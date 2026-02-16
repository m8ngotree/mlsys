## Intro

In a standard transformer, every layer processes every token through the same feed-forward network (FFN). If you make the model bigger by making the FFN bigger, you get more capacity but the amount of compute you have to use scales linearly with the size.

Instead of using one big FFN, you can use many smaller FFNs (called experts), and each token only gets routed to a few of them. This allows you to have massive model capacity (more parameters/capabilities), and it also gives you lower compute per token since each token only activates a subset of experts.

The following is a high-level overview of the MoE pipeline before we dive into specifics later on. You first have your input token (a batch of N tokens). You then have a "router" which decides which K experts the token is routed to by "scoring" every token against each expert (this is called top-K routing). The next step is reordering which groups tokens by destination expert so that compute can be batched efficiently. The expert FFNs then process the tokens that have been assigned to them, and then the tokens are "unsorted" to put them back in their original order. For whatever experts process a token, a weighted combination determines the final output based on the router scores for each expert. 

MoE does have a complex training problem - if the router always sends tokens to just a few experts, the others never get trained/activated and become useless / wasted space. This problem is called load imbalance, and we will eventually discuss the logic that is used to tackle this problem. 

## Routing

The router is a small learned linear layer that outputs a score for every expert based on each token's representation/embedding. A higher score means that the token/expert are more relevant to each other. We first start off with an input of shape [num_tokens, dim] (batch of token embeddings). We then multiply the input against the learned weight matrix in the linear layer which produces one raw score per expert per token. This process is also referred to as a gate projection. We then normalize the scores using either softmax or sigmoid. Softmax makes all expert scores for a given token sum to 1 while sigmoid scores each expert between 0 and 1 independently. Softmax ends up being relative while sigmoid is absolute. Load balancing with softmax is harder since scores are correlated, so models like DeepSeek-V3 use sigmoid.

For each token, we then just keep the K highest-scoring experts. We keep track of the expert # and what its score was. There is also an expert_bias which is added to scores to handle the load balancing problem by artificially boosting underused experts so they get more tokens. However, it is important to note that while the expert_bias impacts the routing decision, it does not impact the weighting as the actual output weighting uses the original scores without the bias. Next, we have the optional process of route normalization where we make the scores of the chosen experts sum to 1 so that the outputs can be combined by weighting and the output scale remains stable. 

There is also the advanced concept of node-limited routing. For large-scale distributed training, tokens can be routed to experts across many machines, which induces a potentially heavy communication cost. Node-limited routing constrains each token to only pick experts from certain node groups. Experts are divided into groups, and group scores are computed for each token (sum of expert scores in each group). The top groups are selected, and all other groups' experts are masked out by setting their scores to -infinity. Then, the normal top-k process is followed within the allowed groups. This keeps communication local to fewer nodes and trades routing quality for faster training.

Below is the code for node-limited routing in torchtitan (TokenChoiceTopKRouter._get_node_limited_routing_scores -models/moe/moe.py line 251):

```
def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
            and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice
```

First, the scores_for_choice scores tensor is reshaped from [num_tokens, num_experts] to [num_tokens, num_expert_groups, experts_per_group] to reorganize the experts into their groups. For each group, the group score is computed by taking the top 2 expert scores within the group and summing them. group_scores has a shape of [num_tokens, num_expert_groups]. Next, the best num_limited_groups are selected for each token using torch.topk(). group_mask is a boolean mask which has the same shape as group_scores - True means that the group is not selected and vice versa for False. The last line modifying scores_for_choice sets the non-selected expert scores to -infinity and reshapes the tensor back to [num_tokens, num_experts] so that only experts from allowed groups can be picked. The following is a walkthrough of this function generated by Claude:

```
2 tokens
6 experts
3 expert groups (so 2 experts per group)
We want to keep only 2 groups per token

Starting tensor: scores_for_choice
Shape [2, 6] — 2 tokens, 6 experts:
token 0: [0.9, 0.1, 0.3, 0.8, 0.2, 0.7]
token 1: [0.1, 0.5, 0.6, 0.2, 0.9, 0.3]

Step 1: .view(-1, num_expert_groups, experts_per_group)

scores_grouped = scores_for_choice.view(-1, 3, 2)

Result shape `[2, 3, 2]`:
token 0: [[0.9, 0.1],   # group 0: experts 0,1
          [0.3, 0.8],   # group 1: experts 2,3
          [0.2, 0.7]]   # group 2: experts 4,5

token 1: [[0.1, 0.5],   # group 0: experts 0,1
          [0.6, 0.2],   # group 1: experts 2,3
          [0.9, 0.3]]   # group 2: experts 4,5

Step 2: .topk(2, dim=-1) then .sum(dim=-1)

top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
group_scores = top2_scores_in_group.sum(dim=-1)

top2_scores_in_group shape [2, 3, 2]:
token 0: [[0.9, 0.1], [0.8, 0.3], [0.7, 0.2]]
token 1: [[0.5, 0.1], [0.6, 0.2], [0.9, 0.3]]

group_scores shape [2, 3] — one score per group per token
token 0: [1.0, 1.1, 0.9]
token 1: [0.6, 0.8, 1.2]

Step 3: torch.topk(group_scores, k=num_limited_groups)
_, group_idx = torch.topk(group_scores, k=2, dim=-1, sorted=False)

token 0: [1, 0]
token 1: [2, 1]

Step 4: building the mask with .scatter_()
group_mask = torch.ones_like(group_scores, dtype=torch.bool)
group_mask.scatter_(1, group_idx, False)

torch.ones_like(group_scores, dtype=torch.bool) creates a boolean tensor of all True, same shape as group_scores — [2, 3]:

[[True, True, True],
 [True, True, True]]

.scatter_(dim, index, value) is the key operation. It says: "for each row (dim=1 means we're scattering along columns), go to the positions given by group_idx and write False there."

For token 0, group_idx = [1, 0] → set columns 1 and 0 to False
For token 1, group_idx = [2, 1] → set columns 2 and 1 to False

Result group_mask shape [2, 3]:

token 0: [False, False, True]    # group 2 is masked (not selected)
token 1: [True, False, False]    # group 0 is masked (not selected)

Step 5: .unsqueeze(-1) and .masked_fill()

scores_for_choice = scores_grouped.masked_fill(
    group_mask.unsqueeze(-1), float("-inf")
).view(-1, self.num_experts)

Problem: group_mask is shape [2, 3] but scores_grouped is shape [2, 3, 2]. You can't apply the mask directly because the dimensions don't align.

.unsqueeze(-1) adds a new dimension at the end: [2, 3] → [2, 3, 1]:

token 0: [[False], [False], [True]]
token 1: [[True],  [False], [False]]

PyTorch then broadcasts this across the last dimension — the [2, 3, 1] mask automatically expands to [2, 3, 2], applying the same mask value to both experts within a group.

.masked_fill(mask, -inf) sets every position where mask is True to -inf.

Result scores_grouped after fill, shape [2, 3, 2]:

token 0: [[0.9, 0.1],      # group 0: kept
          [0.3, 0.8],      # group 1: kept
          [-inf, -inf]]    # group 2: masked out

token 1: [[-inf, -inf],    # group 0: masked out
          [0.6, 0.2],      # group 1: kept
          [0.9, 0.3]]      # group 2: kept


.view(-1, num_experts) reshapes back to [2, 6]:

token 0: [0.9, 0.1, 0.3, 0.8, -inf, -inf]
token 1: [-inf, -inf, 0.6, 0.2, 0.9, 0.3]

```

Below is the code for the main router logic in torchtitan (TokenChoiceTopKRouter.forward - models/moe/moe.py line 293):

```
def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        _, selected_experts_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert
```

The scores = self.gate(x) line is the gate projection. It is matrix multiply projecting from [num_tokens, dim] to [num_tokens, num_experts]  and outputting raw unnormalized scores in each row. We then normalize the scores using sigmoid or softmax in float32 for numerical stability. We then add the load balancing bias if present and optionally implement the node-limited routing. We then select the top-K experts per token using the optionally biased / node-limited scores. Keep in mind that the routing choice is based on scores_for_choice, but the actually scores are taken from the original scores tensor using the gather function based on the chosen indices. This allows the bias / node-limited routing to influence the routing decision but not the output weights. We then optionally implement route normalization to make the routing weights sum to 1. Then, we produce num_tokens_per_experts which is of shape [num_experts] and tracks the number of times each expert was chosen across all tokens and all K selections. This is done by flattening selected_experts_indices into one long list and computing a histogram using torch.histc(). The following is a walkthrough of the forward method generated by Claude:

```
3 tokens
4 experts
top_k = 2
dim = 3

Step 1: scores = self.gate(x)
x shape [3, 3] — 3 tokens, each a vector of size 3:
token 0: [0.5, 0.2, 0.8]
token 1: [0.1, 0.9, 0.3]
token 2: [0.7, 0.4, 0.6]

self.gate(x):

token 0: [1.2, -0.3, 0.8, 0.1]
token 1: [0.4,  0.9, 1.5, 0.2]
token 2: [0.7,  0.3, 0.6, 1.1]

Step 2: sigmoid/softmax normalization

Let's say score_func = "sigmoid". Sigmoid squashes each number independently to (0, 1) via 1 / (1 + e^(-x)):
token 0: [0.77, 0.43, 0.69, 0.52]
token 1: [0.60, 0.71, 0.82, 0.55]
token 2: [0.67, 0.57, 0.65, 0.75]

Step 3: scores_for_choice = scores + expert_bias
Say expert_bias = [0.0, 0.1, -0.1, 0.2] — the load balancer is nudging expert 1 and 3 up, expert 2 down, because experts 1 and 3 have been underused recently.

scores_for_choice shape [3, 4]:
token 0: [0.77, 0.53, 0.59, 0.72]   # expert 1 nudged up, expert 2 nudged down
token 1: [0.60, 0.81, 0.72, 0.75]
token 2: [0.67, 0.67, 0.55, 0.95]

Original scores is untouched — still:
token 0: [0.77, 0.43, 0.69, 0.52]
token 1: [0.60, 0.71, 0.82, 0.55]
token 2: [0.67, 0.57, 0.65, 0.75]

Step 4: torch.topk(scores_for_choice, k=2, dim=-1, sorted=False)

For each token, pick the 2 highest scoring experts from scores_for_choice:
token 0: top scores [0.77, 0.72] → experts [0, 3]
token 1: top scores [0.81, 0.75] → experts [1, 3]
token 2: top scores [0.95, 0.67] → experts [3, 0 or 1]  ← tie between 0 and 1, both 0.67
We discard the scores (_), keeping only selected_experts_indices shape [3, 2]:
[[0, 3],
 [1, 3],
 [3, 0]]

Step 5: top_scores = scores.gather(dim=1, index=selected_experts_indices)

Now we go back to the clean original scores (without bias) and pick out the values at the selected positions.
.gather(dim=1, index=...) says: for each row (token), go to the column positions specified by selected_experts_indices and pull those values out.

Original scores:
token 0: [0.77, 0.43, 0.69, 0.52]   → pick columns 0 and 3 → [0.77, 0.52]
token 1: [0.60, 0.71, 0.82, 0.55]   → pick columns 1 and 3 → [0.71, 0.55]
token 2: [0.67, 0.57, 0.65, 0.75]   → pick columns 3 and 0 → [0.75, 0.67]
top_scores shape [3, 2]:
[[0.77, 0.52],
 [0.71, 0.55],
 [0.75, 0.67]]

Notice: token 0 was routed partly based on the bias nudging expert 3 up, but the output weight for expert 3 is 0.52 — the original unbiased score. The bias did its job (changed the routing decision) but left no fingerprints on the actual computation.

Step 6: route normalization and scaling

denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
top_scores = top_scores / denominator

Sum each token's top_k scores and divide — forces each row to sum to 1.

denominator shape [3, 1] (keepdim=True keeps the dimension so broadcasting works):

[[1.29],
 [1.26],
 [1.42]]

top_scores after normalization:

token 0: [0.77/1.29, 0.52/1.29] = [0.60, 0.40]
token 1: [0.71/1.26, 0.55/1.26] = [0.56, 0.44]
token 2: [0.75/1.42, 0.67/1.42] = [0.53, 0.47]

Each row now sums to 1. Then * self.route_scale multiplies the whole thing by a fixed constant to amplify the weights if needed. This is a hyperparameter tuned during training.

Step 7: torch.histc(...)

num_tokens_per_expert = torch.histc(
    selected_experts_indices.view(-1),
    bins=self.num_experts, min=0, max=self.num_experts
)

.view(-1) flattens selected_experts_indices from [3, 2] to a single list of 6 numbers:

[0, 3, 1, 3, 3, 0]


torch.histc counts how many times each expert index appears, with one bin per expert:

expert 0: appears 2 times  (token 0 and token 2)
expert 1: appears 1 time   (token 1)
expert 2: appears 0 times  (nobody chose it)
expert 3: appears 3 times  (tokens 0, 1, and 2)


num_tokens_per_expert shape [4]:

[2, 1, 0, 3]

This is the load tracking tensor. Expert 3 is overloaded (3 tokens), expert 2 is completely idle. This information will feed into the load balancing system to adjust expert_bias over time by nudging future tokens away from expert 3 and toward expert 2.

The full picture in one pass

x [3,3] → gate → raw scores [3,4] → sigmoid → normalized scores [3,4]
                                                        ↓ (keep clean copy)
                                               + expert_bias → scores_for_choice [3,4]
                                                        ↓
                                                    topk(k=2)
                                                        ↓
                                          selected_experts_indices [3,2]
                                                        ↓
                                    gather from clean scores → top_scores [3,2]
                                                        ↓
                                              normalize + scale
                                                        ↓
                                    histc on flattened indices → load counts [4]

```

## Auxiliary-Loss-Free Load Balancing

As an aside, I will cover how the auxiliary-loss-free load balancing introduced by DeepSeek is implemented here. In the same moe.py file, we have this block of code:

```
self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
```

The expert_bias is a PyTorch buffer of size num_experts that is initialized as all zeros when the load_balance_coeff instance variable is set. The values are used during routing as we discussed previously, but we need to discuss how they are updated. The token counts per expert are accumulated in moe.py lines 497-498 for each forward pass:

```
with torch.no_grad():
    self.tokens_per_expert.add_(num_tokens_per_expert)
```

The biases are updated via an optimizer pre-hook (components/optimizer.py). Before each optimizer step, the pre-hook runs _update_expert_bias(). The main steps are as follows:

```
# All-reduce tokens_per_expert ranks
torch.distributed.all_reduce(tokens_per_expert_by_layer, op=SUM)

# Core update formula:
expert_bias_delta = load_balance_coeff * torch.sign(
    tokens_per_expert.mean() - tokens_per_expert
)
expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
moe.expert_bias.add_(expert_bias_delta)

# Reset accumulator for next optimizer step
moe.tokens_per_expert.zero_()
```

If an expert received fewer tokens than average (mean - actual > 0), this means sign = +1, bias increases, and it becomes more likely to be selected next step. If an expert received more tokens than average ( mean - actual < 0), this means sign = -1, bias decreases, and it becomes less likely to be selected. 

tokens_per_expert.mean() is the ideal — if loads were perfectly balanced, every expert would have exactly this count. tokens_per_expert.mean() - tokens_per_expert computes the gap for each expert. torch.sign(...) reduces each gap to just its direction (-1, 0, or +1). load_balance_coeff (default 1e-3) scales the steps. Subtracting the mean shifts the whole distribution so it sums to zero and prevents biases from drifting in one direction over many steps. zero_() resets the token count accumulator back to zeros for the next round of forward passes. We use torch.sign() instead of the raw magnitude because overloaded experts & noisy token distributions could destabilize training when using raw magnitudes. Using the sign allows for a slower, more stable convergence of the bias.

One interesting thing I found is that torchtitan uses activation checkpointing. Activation checkpointing saves GPU memory by not storing intermediate activations during the forward pass. Instead, during backpropagation it reruns the forward pass from scratch to recompute them. This means forward() runs twice for checkpointed layers. tokens_per_expert.add_(num_tokens_per_expert) is inside forward(), so it runs twice which means the counts are doubled. However, this doesn't matter because doubling all counts doesn't change which experts are above/below the mean. torch.sign() only cares about direction, not magnitude, so this is not an issue either.


## Reordering
After routing, you know which expert each token goes to. However, the assignments might look like this:

```
Token 0 - Expert 3, Expert 7
Token 1 - Expert 0, Expert 2
Token 2 - Expert 3, Expert 1
Token 3 - Expert 0, Expert 5
...
```

Making Expert 3 first process Token 0 and then skip around to find Token 2 is inefficient because of irregular memory access patterns. The solution is to sort/reorder tokens so that all tokens routed to a particular expert are together (token groups are sorted by expert #). The following block shows a sample ordering of a subset of the previous example. It is important to note that the reordered tensor will have N * top_k rows since each of the N tokens is routed to top_k experts.

```
Before reorder:            After reorder:
Token 0 - Expert 3        Token 1 - Expert 0  - Expert 0's tokens start here
Token 1 - Expert 0        Token 3 - Expert 0
Token 2 - Expert 3        Token 2 - Expert 1  - Expert 1's tokens start here
Token 3 - Expert 0        Token 0 - Expert 3  - Expert 3's tokens start here
...                        Token 2 - Expert 3
```

The reordering code logic is implemented in the TokenReorderer.forward() method in line 378 of the same moe.py file. 

```
def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size * seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )
```

The num_tokens_per_expert histogram is computed in the same way as it was in the TokenChoiceTopKRouter forward() method. token_indices_experts_sorted is created by using torch.argsort() on the flattened expert assignment list which returns a list of the indices that would sort the list of tokens by expert # as opposed to sorting the list itself. top_scores_experts_sorted is made by flattening the list of scores and using fancy/advanced indexing through token_indices_experts_sorted to sort the scores in the same reordered sequence such that they correspond to the correct token in token_indices_experts_sorted.

The next part is the MoE.forward() function on line 481:

```
def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # top_scores_experts_sorted and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted // self.router.top_k]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        # Note: we execute the shared expert before scoring the output of the routed expert
        # to "implicitly" overlap the shared expert compute with token combine communication
        out = self.shared_experts(x) if self.shared_experts is not None else None

        # Unsort routed outputs
        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )
        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)

        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)
```

This method calls the router and accumulates token counts for experts as we've discussed previously. It then calls the reorderer so that all tokens going to the same expert are grouped together. It then creates routed_input using this logic: `x[token_indices_experts_sorted // self.router.top_k]`. Before sorting, every (token, expert) pair sits at some position in the flat list. After sorting by expert, we want those pairs in a different order. token_indices_experts_sorted records, for each slot in the new sorted order, which position in the original unsorted list that item came from. If x (the original input) has N rows, token_indices_experts_sorted has N * top_k rows because each token appears once per expert it's routed to. The indexing logic uses the insight that when you flatten [N, top_k] row by row, each token claims exactly top_k consecutive slots. If top_k is 2, Token 0 gets slots 0 and 1, token 1 gets slots 2 and 3, token 2 gets slots 4 and 5. This pattern never changes, so no matter where a flat index ends up after sorting, dividing it by top_k always recovers which token it came from. Thus, routed_input is a tensor of shape [N * top_k, dim] that contains token embeddings duplicated from x and reordered by expert destination. self.score_before_experts determines whether token embeddings are scaled before or after feeding them to the expert. It doesn't seem like there's a clear best option but it's more to just let you experiment.

The next thing to discuss are the two modes for running experts in the FeedForward class. The first is the _run_experts_for_loop method (moe.py line 78):

```
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert_list)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x_splits = torch.split(
        x[: sum(num_tokens_per_expert_list)],
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out
```

The x_splits variable splits tokens by expert based on num_tokens_per_expert_list. The for loop runs a SwiGLU FFN for each expert using its specific weight matrices. The outputs are appended to out_experts_splits. The outputs are the expert-process token embeddings that will be unsorted and weighted. This is the more readable version of expert computation but it's not actually used. The next method is the better version (line 113):

```
def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out
```

This method accomplishes the same task using a GPU kernel for faster computation. The offsets variable stores the end offsets of each expert's chunk in the flattened token list. torch._grouped_mm is a single fused kernel call that runs one matmul per expert group in an optimized manner. There is also logic in models/moe/utils.py that handles padding for this, as each group's token count needs to be a multiple of 8.

Lastly, we have the unsorting logic in MoE.forward() (line 531).

```
# Unsort routed outputs
        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )
        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)

        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)
```

routed_output_unsorted is the destination tensor that will store the original token order (routed_output contains expert outputs in expert-sorted order). The next line uses a scatter operation to write each row in routed_output to the slot it originally came from. It is then reshaped to [N, top_k, dim] so each token has a slice containing top_k rows (one for each expert it visited). There are two paths: if routing scores haven't been applied yet then the code executes a batched matrix multiply to get the weighted sum of each expert's output with its routing score. If the scores were already applied before the experts ran, then you just sum along the top_k dimension.

There can be optional shared experts which are dense FFNs that every token passes through regardless of routing. It's like a guaranteed baseline computation on top of the sparse routed computation. 
It can run in parallel with the unsorting operation, as unsorting may involve communication across devices. Computing the shared experts results can overlap compute with the communication latency to optimize performance. The out variable is the shared experts, so if it is None then we just return out_experts in its original shape - otherwise we add the shared experts to out_experts and reshape to the original shape.

## Load Balancing
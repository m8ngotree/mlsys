## Intro

When training neural networks, optimizers use the computed gradients to update parameters/weights in the direction that most reduces the loss. Memory is a factor for optimizers like Adam as it maintains two extra states per parameter: momentum & variance. Momentum is a running average of the past gradients for a parameter, and variance is a running average of the past squared gradients for a parameter. Thus, when training neural networks, the following four things need to be stored for each parameter: the parameter/weight (tensor of floating point value), gradient, momentum, & variance. If a model has 10 billion parameters and every value is stored in 32-bit format, it would need 160 GB just for the parameters and optimizer states which exceeds the capacity of an A100 GPU (80 GB). This doesn't even factor in the memory needed for intermediate activations during the forward and backward passes. ZeRO is intended to mitigate this problem.

## Gradients
Gradients denote how much a small change in a parameter would affect the total loss. In mathematical terms, it is the partial derivative of the loss functon with respect to a parameter. The gradient points in the direction of the greatest rate of increase of the loss function, so we adjust parameters in the opposite direction of the gradient. Asking Claude for a mathematical walkthrough of gradients in a simple neural network can help understand the basic process and gradient accumulation.

## Data Parallelism
ZeRO is an improvement on the naive approach of data parallelism, so we should understand this first. Data parallelism involves using multiple GPUs and splitting up the input data across each GPU so that each GPU conducts forward/backward passes on its slice of the data. Each GPU has a full copy of the model which makes this possible. Data parallelism drastically increases throughput as forward/backward passes on different data can be done in parallel. It does not really offer memory savings as each GPU has to have a copy of the model, gradients, and optimizer states. 

Data parallelism is enabled by averaging gradients across GPUs. After each GPU does its forward/backward pass and computes gradients, each GPU will have slightly different gradients as they processed different data (they usually process equal amounts of data in order to facilitate a simple averaging of gradients - if they processed different amounts they would have to do a weighted average which could get complicated). The GPUs must use a communication collective operation called AllReduce in order to average the gradients and ensure each model has the same copy of the averaged gradients. In practice, an approach called Ring AllReduce is used to average gradients for distributed training and ensure each model has the averaged gradients. Asking Claude for a detailed walkthrough of Ring AllReduce is a good way to learn, but I'll give one below (the numbers in the example are made with Claude).

## Ring AllReduce

In Ring AllReduce, each GPU has its gradients which need to be averaged and each GPU needs to finish the process holding the averaged gradients in memory. There are two phases: the first is Reduce-Scatter (compute partial sums and distribute) and the second is AllGather (collect partial sums across all GPUs), both of which are communication collective operations. Let's assume there are 4 GPUs (0, 1, 2, 3) and the model has 8 parameters (so there are 8 gradients). 

```
GPU 0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
GPU 1: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
GPU 2: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
GPU 3: [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

Average: [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75]
```

Each GPU divides its gradients into 4 chunks (one per GPU / 2 gradients per chunk). 

```
GPU 0: [1.0, 2.0] | [3.0, 4.0] | [5.0, 6.0] | [7.0, 8.0]
        Chunk 0      Chunk 1      Chunk 2      Chunk 3

GPU 1: [2.0, 3.0] | [4.0, 5.0] | [6.0, 7.0] | [8.0, 9.0]
        Chunk 0      Chunk 1      Chunk 2      Chunk 3

GPU 2: [1.5, 2.5] | [3.5, 4.5] | [5.5, 6.5] | [7.5, 8.5]
        Chunk 0      Chunk 1      Chunk 2      Chunk 3

GPU 3: [2.5, 3.5] | [4.5, 5.5] | [6.5, 7.5] | [8.5, 9.5]
        Chunk 0      Chunk 1      Chunk 2      Chunk 3
```

The GPUs are visualized as a ring where 0 connects to 1, 1 connects to 2, 2 connects to 3, and 3 connects back to 0 (assume connections are clockwise). Each GPU sends info to its clockwise neighbor and receives from its counterclockwise neighbor. For the first iteration, GPU i sends its chunk i to its clockwise neighbor (e.g. GPU 0 sends its chunk 0 to GPU 1). Thus, each GPU is receiving a chunk from its counterclockwise neighbor. Each GPU takes the chunk it received and adds it to its own corresponding chunk - it is important to keep in mind that the chunks that are added on a GPU depends on who it received from. Since GPU 3 sends its chunk 3 to GPU 0, GPU 0 adds its chunk 3 to the chunk 3 it received from GPU 3 (this logic applied to all GPUs). 

```
GPU 0 sends Chunk 0 [1.0, 2.0] to GPU 1
GPU 1 sends Chunk 1 [4.0, 5.0] to GPU 2
GPU 2 sends Chunk 2 [5.5, 6.5] to GPU 3
GPU 3 sends Chunk 3 [8.5, 9.5] to GPU 0

GPU 0: Chunk 3 = [7.0, 8.0] + [8.5, 9.5] = [15.5, 17.5]
GPU 1: Chunk 0 = [2.0, 3.0] + [1.0, 2.0] = [3.0, 5.0]
GPU 2: Chunk 1 = [3.5, 4.5] + [4.0, 5.0] = [7.5, 9.5]
GPU 3: Chunk 2 = [6.5, 7.5] + [5.5, 6.5] = [12.0, 14.0]
```

After the first iteration, each GPU has the partial sum of 2 GPUs for one chunk. In the next iteration, each GPU sends its chunk that was updated to the next GPU (i.e. GPU 0 sends its chunk 3 to GPU 1, GPU 1 sends its chunk 0 to GPU 2, etc.). 

```
GPU 0 sends updated Chunk 3 [15.5, 17.5] to GPU 1
GPU 1 sends updated Chunk 0 [3.0, 5.0] to GPU 2
GPU 2 sends updated Chunk 1 [7.5, 9.5] to GPU 3
GPU 3 sends updated Chunk 2 [12.0, 14.0] to GPU 0
```

Each GPU adds the chunk it received to its corresponding chunk (i.e. GPU 0 received Chunk 2 from GPU 3 so it adds what it received to its own Chunk 2).

```
GPU 0: Chunk 2 = [5.0, 6.0] + [12.0, 14.0] = [17.0, 20.0]
GPU 1: Chunk 3 = [8.0, 9.0] + [15.5, 17.5] = [23.5, 26.5]
GPU 2: Chunk 0 = [1.5, 2.5] + [3.0, 5.0] = [4.5, 7.5]
GPU 3: Chunk 1 = [4.5, 5.5] + [7.5, 9.5] = [12.0, 15.0]
```

After the second iteration, each GPU has the partial sum of 3 GPUs for one chunk. In the last iteration (there are numGPUs - 1 iterations), the same behavior continues (each GPU sends its chunk that was last updated to the next GPU). 

```
GPU 0 sends Chunk 2 [17.0, 20.0] to GPU 1
GPU 1 sends Chunk 3 [23.5, 26.5] to GPU 2
GPU 2 sends Chunk 0 [4.5, 7.5] to GPU 3
GPU 3 sends Chunk 1 [12.0, 15.0] to GPU 0

GPU 0: Chunk 1 = [3.0, 4.0] + [12.0, 15.0] = [15.0, 19.0] ✓ (sum of all 4 GPUs for Chunk 1)
GPU 1: Chunk 2 = [6.0, 7.0] + [17.0, 20.0] = [23.0, 27.0] ✓ (sum of all 4 GPUs for Chunk 2)
GPU 2: Chunk 3 = [7.5, 8.5] + [23.5, 26.5] = [31.0, 35.0] ✓ (sum of all 4 GPUs for Chunk 3)
GPU 3: Chunk 0 = [2.5, 3.5] + [4.5, 7.5] = [7.0, 11.0] ✓ (sum of all 4 GPUs for Chunk 0)
```

Thus, after the Reduce-Scatter phase, each GPU holds the chunk corresponding to the number of its clockwise neighbor (GPU 0 holds chunk 1, GPU 1 holds chunk 2, etc.).

```
GPU 0: [?, ?] | [15.0, 19.0] | [?, ?] | [?, ?]
GPU 1: [?, ?] | [?, ?] | [23.0, 27.0] | [?, ?]
GPU 2: [?, ?] | [?, ?] | [?, ?] | [31.0, 35.0]
GPU 3: [7.0, 11.0] | [?, ?] | [?, ?] | [?, ?]
```

In the next phase (AllGather), each GPU communicates a final chunk to its clockwise neighbor in each iteration. In the first iteration, each GPU sends its completed chunk.

```
GPU 0 sends Chunk 1 [15.0, 19.0] to GPU 1
GPU 1 sends Chunk 2 [23.0, 27.0] to GPU 2
GPU 2 sends Chunk 3 [31.0, 35.0] to GPU 3
GPU 3 sends Chunk 0 [7.0, 11.0] to GPU 0
```

Each GPU simply writes the chunk it receives in the correct position.

```
GPU 0: [7.0, 11.0] | [15.0, 19.0] | [?, ?] | [?, ?]
GPU 1: [?, ?] | [15.0, 19.0] | [23.0, 27.0] | [?, ?]
GPU 2: [?, ?] | [?, ?] | [23.0, 27.0] | [31.0, 35.0]
GPU 3: [7.0, 11.0] | [?, ?] | [?, ?] | [31.0, 35.0]
```

Iteration 2 (GPU sends chunk it just received for all iterations in All-Gather):

```
GPU 0 sends Chunk 0 [7.0, 11.0] to GPU 1
GPU 1 sends Chunk 1 [15.0, 19.0] to GPU 2
GPU 2 sends Chunk 2 [23.0, 27.0] to GPU 3
GPU 3 sends Chunk 3 [31.0, 35.0] to GPU 0

GPU 0: [7.0, 11.0] | [15.0, 19.0] | [?, ?] | [31.0, 35.0]
GPU 1: [7.0, 11.0] | [15.0, 19.0] | [23.0, 27.0] | [?, ?]
GPU 2: [?, ?] | [15.0, 19.0] | [23.0, 27.0] | [31.0, 35.0]
GPU 3: [7.0, 11.0] | [?, ?] | [23.0, 27.0] | [31.0, 35.0]
```

Iteration 3:

```
GPU 0 sends Chunk 3 [31.0, 35.0] to GPU 1
GPU 1 sends Chunk 0 [7.0, 11.0] to GPU 2
GPU 2 sends Chunk 1 [15.0, 19.0] to GPU 3
GPU 3 sends Chunk 2 [23.0, 27.0] to GPU 0

GPU 0: [7.0, 11.0] | [15.0, 19.0] | [23.0, 27.0] | [31.0, 35.0]
GPU 1: [7.0, 11.0] | [15.0, 19.0] | [23.0, 27.0] | [31.0, 35.0]
GPU 2: [7.0, 11.0] | [15.0, 19.0] | [23.0, 27.0] | [31.0, 35.0]
GPU 3: [7.0, 11.0] | [15.0, 19.0] | [23.0, 27.0] | [31.0, 35.0]
```

Each GPU divides by the number of GPU to get the average gradients.

```
All GPUs: [7.0/4, 11.0/4] | [15.0/4, 19.0/4] | [23.0/4, 27.0/4] | [31.0/4, 35.0/4]
        = [1.75, 2.75] | [3.75, 4.75] | [5.75, 6.75] | [7.75, 8.75]
```

This is better than the naive approach where one GPU receives everything from all other GPUs and then broadcasts it because the initial gather would be a bottleneck. Ring AllReduce allows the communication to happen in parallel which speeds up the total communication time.

## ZeRO Stage 1 - Partition Optimizer States

In the standard data parallelism approach described above, each GPU contains the same averaged gradients after the AllReduce. Each GPU uses all optimizer states to calculate all updates for all parameters. It is clear that they are doing redundant work and storing redundant states. The core optimization of ZeRO Stage 1 is that we can partition the optimizer states across GPUs. Each partition of optimizer states can be used to calculate the update for the respective set of parameters, and the results can be communicated across GPUs. If we have N GPUs, the naive data parallelism approach stores N parameters, gradients, momentum, & variance states per GPU. The Stage 1 optimization stores N parameters & gradients per GPU, but 1/N momentum and variance states per GPU. This results in total memory savings of ~25-50%, depending on the number of GPUs. 

The Stage 1 optimization uses the same forward and backward pass as before - each GPU has the full model, each GPU does a forward pass on its data slice, & each GPU computes its gradients during its backward pass. The difference starts in the gradient sync. In standard data parallelism, the full AllReduce results in each GPU having all the averaged gradients. In ZeRO Stage 1, a Reduce-Scatter operation is done to only give each GPU its partition of the averaged gradients. In the optimizer step, each GPU updates its assigned parameters (based on the gradient partition it has) by using its partition of optimizer states. After this, each GPU has a partition of updated parameters which need to be communicated to the other GPUs. The communication of all updated parameters to all GPUs is done with an AllGather. Thus, we are saving memory with basically the same communication cost as the standard data parallelism approach (1 Reduce-Scatter + 1 AllGather).

## Stage 1 Code

In the DeepSpeed repo, lines 212-213 of the *deepspeed/runtime/zero/stage_1_and_2.py* file contains some code relevant to Stage 1.

```
self.partition_gradients = partition_grads
self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"
```

Partitioning gradients is a feature in ZeRO Stage 2, so not including this optimization denotes that the code will use Stage 1.

```
data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].detach().clone().to(
    device=self.device, dtype=self.master_weights_and_grads_dtype)

self.single_partition_of_fp32_groups.append(weights_partition)

param_group['params'] = [self.single_partition_of_fp32_groups[i]]
``` 

## Adam Optimizer
As a short aside, I'll explain how the Adam optimizer works.

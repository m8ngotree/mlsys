import torch
import torch.nn as nn

m, n = 1024, 1024
num_warmup = 10
num_iterations = 1000

input = torch.arange(1, m * n + 1).reshape(m, n).float().cuda()

layer_norm = nn.LayerNorm(n, elementwise_affine=False, eps=1e-6).cuda()

for _ in range(num_warmup):
    output = layer_norm(input)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(num_iterations):
    output = layer_norm(input)
end_event.record()

torch.cuda.synchronize()

total_time_ms = start_event.elapsed_time(end_event)
avg_time_ms = total_time_ms / num_iterations

print(f"Total time: {total_time_ms:.4f} ms")
print(f"Average time per iteration: {avg_time_ms:.4f} ms")
print(f"Output mean (should be ~0): {output[0].mean().item():.6f}")
print(f"Output std (should be ~1): {output[0].std().item():.6f}")
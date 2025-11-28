import torch
import time

N = 128 * 1024 * 1024
device = 'cuda'

x = torch.randn(N, dtype=torch.float16, device=device) * 2.0

for _ in range(10):
    scale = x.abs().max() / 127.0
    x_quant = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)

torch.cuda.synchronize()
start = time.time()
num_iters = 100

for _ in range(num_iters):
    scale = x.abs().max() / 127.0
    x_quant = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)

torch.cuda.synchronize()
end = time.time()

avg_time_ms = (end - start) * 1000 / num_iters
bytes_transferred = N * 2 + N * 1
bandwidth_gbs = (bytes_transferred / (avg_time_ms / 1000)) / 1e9

print(f"PyTorch Quantization:")
print(f"Time: {avg_time_ms:.3f} ms")
print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
print(f"\nYour best kernel: 774 GB/s")
print(f"Speedup: {774/bandwidth_gbs:.2f}x")
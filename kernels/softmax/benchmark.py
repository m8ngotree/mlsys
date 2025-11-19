# https://maharshi.bearblog.dev/optimizing-softmax-cuda/

import torch
import time
import torch.nn.functional as F

matrix = torch.randn(1024, 32768, device='cuda', dtype=torch.float32)

_ = F.softmax(matrix, dim=-1)

torch.cuda.synchronize()

total_time = 0
n_iters = 5

for _ in range(n_iters):
    torch.cuda.synchronize()
    start = time.time()
    _ = F.softmax(matrix, dim=-1)
    torch.cuda.synchronize()
    end = time.time()

    total_time += (end - start) * 1000
    print(total_time)

print(f"Softmax computation time (average): {(total_time / n_iters):.3f} ms")
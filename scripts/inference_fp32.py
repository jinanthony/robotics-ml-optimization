import time
import torch
from ultralytics import YOLO

device = "cuda"
model = YOLO("yolov8n.pt").to(device)
dummy = torch.randn(1, 3, 640, 640).to(device)

# Warmup
for _ in range(30):
    model(dummy)

torch.cuda.synchronize()

N = 200
latencies = []
for _ in range(N):
    start = time.perf_counter()
    model(dummy)
    torch.cuda.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)

print(f"FP32 mean latency: {sum(latencies)/len(latencies):.2f} ms")
print(f"P95 latency: {sorted(latencies)[int(0.95 * len(latencies))]:.2f} ms")
print(f"GPU mem: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

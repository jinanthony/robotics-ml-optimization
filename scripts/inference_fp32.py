import time
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

dummy = torch.randn(1, 3, 640, 640).to(device)

# Warmup
for _ in range(10):
    model(dummy)

torch.cuda.synchronize()
start = time.time()

N = 100
for _ in range(N):
    model(dummy)

torch.cuda.synchronize()
end = time.time()

latency_ms = (end - start) * 1000 / N
mem_mb = torch.cuda.max_memory_allocated() / 1024**2

print(f"FP32 Latency: {latency_ms:.2f} ms")
print(f"GPU Memory: {mem_mb:.1f} MB")

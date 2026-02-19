import time
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("yolov8n_int8.onnx", providers=["CUDAExecutionProvider"])
input_name = sess.get_inputs()[0].name

dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Warmup
for _ in range(10):
    sess.run(None, {input_name: dummy})

start = time.time()
N = 100
for _ in range(N):
    sess.run(None, {input_name: dummy})
end = time.time()

latency_ms = (end - start) * 1000 / N
print(f"INT8 ONNX Latency: {latency_ms:.2f} ms")

# Real-Time Robotics Inference Optimization

This project demonstrates ML infrastructure optimization techniques for real-time robotics perception on GPU:
- Baseline FP32 inference
- INT8 quantization via ONNX Runtime
- Structured pruning
- GPU profiling and benchmarking
- Real-time inference loop

## Setup
```bash
conda create -n ml-infra python=3.10
conda activate ml-infra
pip install -r requirements.txt

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=12)
print("Exported yolov8n.onnx")

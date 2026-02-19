import os
import cv2
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class CalibDataReader(CalibrationDataReader):
    def __init__(self, img_dir="calib_images"):
        self.img_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.idx = 0

    def get_next(self):
        while self.idx < len(self.img_paths):
            path = self.img_paths[self.idx]
            self.idx += 1

            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Skipping unreadable image: {path}")
                continue

            img = cv2.resize(img, (640, 640))
            img = img.transpose(2, 0, 1) / 255.0
            img = img[np.newaxis, :].astype(np.float32)

            return {"images": img}

        return None

quantize_static(
    model_input="yolov8n.onnx",
    model_output="yolov8n_int8.onnx",
    calibration_data_reader=CalibDataReader(),
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)

print("Saved yolov8n_int8.onnx")

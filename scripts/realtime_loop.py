import cv2
import time
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt").cuda()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame)
    latency = (time.time() - start) * 1000

    cv2.putText(frame, f"{latency:.1f} ms",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Robotics Inference", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

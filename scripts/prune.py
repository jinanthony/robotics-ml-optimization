import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

model = YOLO("yolov8n.pt").model

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
        prune.remove(module, 'weight')

torch.save(model.state_dict(), "yolov8n_pruned.pt")
print("Saved pruned model weights")

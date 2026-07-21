import os
import torch
from ultralytics import YOLO

def main() -> None:
    model_name = "yolo11m.pt"
    model = YOLO(model_name)

    if torch.cuda.is_available():
        device = "0"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Warning: No GPU detected, falling back to CPU")

    model.train(
        data="dataset/data.yaml",
        epochs=200,
        imgsz=1280,
        batch=4, # Use 4 batch size to prevent OOM with large image size
        device=device,
        optimizer="SGD",
        lr0=0.01,
        project="models",
        name="tyre_merged",
        exist_ok=True,
        patience=30,
        workers=0,
        cache=False,
    )

if __name__ == "__main__":
    main()

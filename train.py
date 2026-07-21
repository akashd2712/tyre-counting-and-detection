import os
import torch
from ultralytics import YOLO


def main() -> None:
    model_name = "yolo11n.yaml"
    if os.path.exists("yolo11n.pt"):
        model_name = "yolo11n.pt"
    model = YOLO(model_name)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    batch_size = 2 if device == "cpu" else 8
    image_size = 416 if device == "cpu" else 640
    model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        project="models",
        name="tyre_yolo11",
        exist_ok=True,
        patience=20,
        augment=True,
        workers=0,
        cache=False,
    )


if __name__ == "__main__":
    main()

import os
import torch
from ultralytics import YOLO


def main() -> None:
    model_name = "yolo11n.yaml"
    if os.path.exists("yolo11n.pt"):
        model_name = "yolo11n.pt"
    model = YOLO(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,   # RTX 3050 4 GB VRAM – keep at 8 to avoid OOM
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        project="models",
        name="tyre_yolo11",
        exist_ok=True,
        patience=20,
        augment=True,
        workers=0,     # Windows fix: avoids multiprocessing paging file exhaustion
    )


if __name__ == "__main__":
    main()

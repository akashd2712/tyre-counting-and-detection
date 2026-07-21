from __future__ import annotations

import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gather_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    source_dir = root / "dataset" / "images"
    train_images_dir = source_dir / "train"
    val_images_dir = source_dir / "val"
    train_labels_dir = root / "dataset" / "labels" / "train"
    val_labels_dir = root / "dataset" / "labels" / "val"

    for directory in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    images = gather_images(source_dir)
    images = [path for path in images if path.parent == source_dir]

    if not images:
        print("No images were found in dataset/images.")
        return

    split_index = max(1, int(len(images) * 0.9))
    train_images = images[:split_index]
    val_images = images[split_index:]

    for image_path in train_images:
        target_image = train_images_dir / image_path.name
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
        label_path = train_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")

    for image_path in val_images:
        target_image = val_images_dir / image_path.name
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
        label_path = val_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")

    print(f"Prepared {len(train_images)} training images and {len(val_images)} validation images.")


if __name__ == "__main__":
    main()

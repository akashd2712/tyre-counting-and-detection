"""Stitch dataset/images/frame_*.jpg into a video for the counting pipeline."""
import glob
import os
import cv2
import re

IMG_DIR = r"A:\tyre\dataset\images"
OUT_PATH = r"A:\tyre\dataset\dataset_video.mp4"
FPS = 10  # 10 fps gives a natural speed for 444 frames (~44 seconds)


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(s))]


def main():
    pattern = os.path.join(IMG_DIR, "frame_[0-9]*.jpg")
    files = sorted(glob.glob(pattern), key=natural_sort_key)
    print(f"Found {len(files)} frames")

    if not files:
        print("No frames found!")
        return

    sample = cv2.imread(files[0])
    h, w = sample.shape[:2]
    print(f"Resolution: {w}x{h}")

    writer = cv2.VideoWriter(
        OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h),
    )

    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            print(f"  Skipping unreadable: {f}")
            continue
        writer.write(img)
        if (i + 1) % 50 == 0:
            print(f"  Written {i+1}/{len(files)} frames")

    writer.release()
    print(f"Done → {OUT_PATH}  ({len(files)} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()

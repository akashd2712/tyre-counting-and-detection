# Smart Tyre Detection and Bidirectional Counting

This project implements a high-accuracy tyre counting pipeline using YOLOv11 object detection, tracking, and a virtual counting line.

## Features
- Detects tyres in live CCTV or RTSP video feeds
- Ignores people and unrelated motion by focusing on tyre detections only
- Uses a virtual grid line to count entry and exit events
- Prevents double counting with object tracking
- Supports ROI cropping to focus on the container opening

## Structure
- app.py: main inference pipeline
- train.py: training entry point for custom tyre datasets
- counting/: counting and direction logic
- utils/: ROI, motion filtering, and visualization helpers

## Run
1. Install dependencies: pip install -r requirements.txt
2. Place a trained YOLOv11 model in models/best.pt or use a pretrained model automatically.
3. Run: python app.py --source rtsp://your-camera --show

## Training
1. Prepare your dataset in dataset/images/train, dataset/images/val, and dataset/images/labels
2. Update dataset/data.yaml with your class names
3. Run: python train.py

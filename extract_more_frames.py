import cv2
import os
from pathlib import Path

# Paths
INPUT_DIR = Path("A:/videos")
OUTPUT_DIR = Path("A:/tyre/dataset/images")

# Settings
FRAME_INTERVAL_SEC = 5.0  # Extract 1 frame every 5 seconds
MAX_FRAMES_PER_VIDEO = 50 # Prevent generating tens of thousands of frames from 16GB videos

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all video files
    video_files = []
    for ext in ["*.mov", "*.mp4", "*.avi"]:
        video_files.extend(INPUT_DIR.rglob(ext))
    
    if not video_files:
        print(f"No videos found in {INPUT_DIR}")
        return

    print(f"Found {len(video_files)} videos. Starting extraction...")
    total_extracted = 0

    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"  [!] Could not open video: {video_path.name}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps: # Handle invalid FPS
            fps = 30.0 
            
        frame_interval_count = int(fps * FRAME_INTERVAL_SEC)
        
        frame_idx = 0
        extracted_count = 0
        
        while extracted_count < MAX_FRAMES_PER_VIDEO:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create a unique filename for this frame
            safe_name = video_path.stem.replace(" ", "_").replace(".", "_")
            out_filename = f"{safe_name}_f{frame_idx}.jpg"
            out_path = OUTPUT_DIR / out_filename
            
            # Save the frame
            cv2.imwrite(str(out_path), frame)
            
            extracted_count += 1
            total_extracted += 1
            frame_idx += frame_interval_count
            
            if extracted_count % 10 == 0:
                print(f"  Extracted {extracted_count} frames so far...")
                
        cap.release()
        print(f"  Done with {video_path.name}: Extracted {extracted_count} frames.")
        
    print(f"\nExtraction complete! Successfully added {total_extracted} new frames to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

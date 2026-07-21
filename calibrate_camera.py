import argparse
import sys
import cv2
import numpy as np
import yaml

points = []
frame_copy = None

def click_event(event, x, y, flags, param):
    global points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame_copy, str(len(points)), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if len(points) > 1:
                cv2.line(frame_copy, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(frame_copy, points[-1], points[0], (0, 255, 0), 2)
            cv2.imshow("Camera Calibration", frame_copy)

def main():
    parser = argparse.ArgumentParser(description="Calibrate Homography for Tyre Counter")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load config to get video source
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    source = config.get("source", {}).get("input", "0")
    print(f"Opening video source: {source}")

    import pathlib
    source_path = pathlib.Path(source)
    if source_path.is_dir():
        # Get first video file
        extensions = tuple(config.get("source", {}).get("extensions", [".mov", ".mp4", ".avi", ".mkv"]))
        playlist = sorted(p for p in source_path.iterdir() if p.suffix.lower() in extensions)
        if not playlist:
            print(f"Error: No videos found in {source}")
            sys.exit(1)
        cap = cv2.VideoCapture(str(playlist[0]))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read a frame from the video source.")
        sys.exit(1)

    global frame_copy
    frame_copy = frame.copy()

    print("\n--- CAMERA CALIBRATION ---")
    print("Click 4 points on the floor that form a RECTANGLE in real life.")
    print("Order matters! Click them in this order:")
    print("  1. Top-Left")
    print("  2. Top-Right")
    print("  3. Bottom-Right")
    print("  4. Bottom-Left")
    print("\nPress 'c' to clear points. Press 'q' or 'ESC' to quit without saving.")

    cv2.imshow("Camera Calibration", frame_copy)
    cv2.setMouseCallback("Camera Calibration", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            print("Calibration cancelled.")
            sys.exit(0)
        elif key == ord('c'):
            points.clear()
            frame_copy = frame.copy()
            cv2.imshow("Camera Calibration", frame_copy)
            print("Points cleared.")
        elif len(points) == 4:
            print("\n4 points selected. Press 'ENTER' or 'SPACE' to confirm, or 'c' to clear.")
            key = cv2.waitKey(0) & 0xFF
            if key == 13 or key == 32:  # Enter or Space
                break
            elif key == ord('c'):
                points.clear()
                frame_copy = frame.copy()
                cv2.imshow("Camera Calibration", frame_copy)

    cv2.destroyAllWindows()

    src_pts = np.array(points, dtype="float32")
    
    # Destination points: a top-down bird's-eye view.
    # Let's map it to a square of 500x500 pixels.
    width_td = 500
    height_td = 500
    dst_pts = np.array([
        [0, 0],
        [width_td - 1, 0],
        [width_td - 1, height_td - 1],
        [0, height_td - 1]
    ], dtype="float32")

    M, status = cv2.findHomography(src_pts, dst_pts)
    
    if M is None:
        print("Error: Could not compute Homography Matrix. Try picking a more regular shape.")
        sys.exit(1)

    # Save to config
    # Convert numpy array to list for yaml serialization
    matrix_list = M.tolist()
    
    config.setdefault("homography", {})
    config["homography"]["enabled"] = True
    config["homography"]["matrix"] = matrix_list
    config["homography"]["src_points"] = src_pts.tolist()
    config["homography"]["map_width"] = width_td
    config["homography"]["map_height"] = height_td

    with open(args.config, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print("\n✅ Calibration successful! Homography matrix saved to config.yaml.")
    print("You can now run app.py with top-down 3D perspective tracking.")

if __name__ == "__main__":
    main()

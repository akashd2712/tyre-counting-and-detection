import ultralytics
from pathlib import Path

print('ultralytics', ultralytics.__version__)
root = Path(ultralytics.__file__).resolve().parent
for path in sorted(root.glob('cfg/models/**/*.yaml')):
    if 'yolo' in path.name.lower() or 'yolov' in path.name.lower():
        print(path.relative_to(root))

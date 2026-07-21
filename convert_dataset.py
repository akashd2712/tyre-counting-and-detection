import json
import os
import shutil
from pathlib import Path

TARGET_CLASSES = {
    'moving_tyre': 0,
    'stationary_tyre': 1,
    'tyre_group': 2,
    'person': 3,
    'loading_vehicle': 4,
}

def get_class_id(coco_name):
    if coco_name == 'Tyre':
        return 1 # treat generic Tyre as stationary_tyre
    return TARGET_CLASSES.get(coco_name, -1)

def convert(split_name, coco_dir, yolo_images_dir, yolo_labels_dir):
    coco_json = Path(coco_dir) / f"{split_name}" / "_annotations.coco.json"
    if not coco_json.exists():
        print(f"Skipping {split_name}, JSON not found.")
        return
    with open(coco_json) as f:
        data = json.load(f)
    
    categories = {c['id']: c['name'] for c in data.get('categories', [])}
    images = {img['id']: img for img in data.get('images', [])}
    
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    from collections import defaultdict
    annotations_by_img = defaultdict(list)
    for ann in data.get('annotations', []):
        annotations_by_img[ann['image_id']].append(ann)
        
    for img_id, img_info in images.items():
        img_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        src_img = Path(coco_dir) / f"{split_name}" / img_name
        dst_img = Path(yolo_images_dir) / img_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            
        label_file = Path(yolo_labels_dir) / (Path(img_name).stem + ".txt")
        with open(label_file, "w") as f:
            for ann in annotations_by_img.get(img_id, []):
                cat_name = categories[ann['category_id']]
                yolo_class_id = get_class_id(cat_name)
                if yolo_class_id == -1:
                    continue
                x, y, w, h = ann['bbox']
                cx = (x + w/2) / img_width
                cy = (y + h/2) / img_height
                nw = w / img_width
                nh = h / img_height
                f.write(f"{yolo_class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    print(f"Finished {split_name}.")

if __name__ == '__main__':
    convert('train', 'Tyre.coco', 'dataset/images/train', 'dataset/labels/train')
    convert('valid', 'Tyre.coco', 'dataset/images/val', 'dataset/labels/val')
    convert('test', 'Tyre.coco', 'dataset/images/test', 'dataset/labels/test')

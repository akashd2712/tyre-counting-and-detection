import os
from pathlib import Path

# Original classes from Roboflow:
# 0: Tyre
# 1: loading_vehicle
# 2: moving_tyre
# 3: person
# 4: stationary_tyre
# 5: tyre_group
# 6: weqeq

# Mapping to new classes:
# 0: tyre (merged: Tyre, stationary_tyre, tyre_group)
# 1: moving_tyre
# 2: loading_vehicle
# 3: person
# (weqeq is deleted)

MAPPING = {
    "0": "0",
    "4": "0",
    "5": "0",
    "2": "1",
    "1": "2",
    "3": "3",
}

def remap_dir(label_dir):
    p = Path(label_dir)
    if not p.exists():
        return
    
    total_files = 0
    total_lines = 0
    deleted_lines = 0
    mapped_lines = 0
    
    for txt_file in p.glob("*.txt"):
        total_files += 1
        with open(txt_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            parts = line.split()
            cls_id = parts[0]
            
            if cls_id in MAPPING:
                parts[0] = MAPPING[cls_id]
                new_lines.append(" ".join(parts) + "\n")
                mapped_lines += 1
            else:
                deleted_lines += 1
                
        with open(txt_file, "w") as f:
            f.writelines(new_lines)
            
    print(f"{label_dir}: processed {total_files} files.")
    print(f"  - Total boxes before: {total_lines}")
    print(f"  - Boxes re-mapped: {mapped_lines}")
    print(f"  - Boxes deleted: {deleted_lines}")

def main():
    splits = ["dataset/train/labels", "dataset/valid/labels", "dataset/test/labels"]
    for s in splits:
        remap_dir(s)

    # Update data.yaml
    data_yaml = Path("dataset/data.yaml")
    if data_yaml.exists():
        content = """train: train/images
val: valid/images
test: test/images

nc: 4
names: ['tyre', 'moving_tyre', 'loading_vehicle', 'person']
"""
        with open(data_yaml, "w") as f:
            f.write(content)
        print("Updated dataset/data.yaml with new classes.")

if __name__ == "__main__":
    main()

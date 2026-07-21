import os
from pathlib import Path

# Mapping rules: 
# 0, 2, 4, 5 -> 0
# 1, 3, 6 -> delete
VALID_CLASSES = {"0", "2", "4", "5"}

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
            
            if cls_id in VALID_CLASSES:
                # Map to 0
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")
                mapped_lines += 1
            else:
                deleted_lines += 1
                
        with open(txt_file, "w") as f:
            f.writelines(new_lines)
            
    print(f"{label_dir}: processed {total_files} files.")
    print(f"  - Total boxes before: {total_lines}")
    print(f"  - Boxes re-mapped to 0: {mapped_lines}")
    print(f"  - Boxes deleted: {deleted_lines}")

def main():
    splits = ["dataset/train/labels", "dataset/valid/labels", "dataset/test/labels"]
    for s in splits:
        remap_dir(s)

if __name__ == "__main__":
    main()

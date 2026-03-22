#!/usr/bin/env python3
import os
from pathlib import Path

ROOT = Path("/home/rs_students/advstr/dataset/RoadText1k/output_rainy")  # adjust if needed

def make_gt(split: str):
    split_root = ROOT / split
    gt_path = Path(f"{split}_gt.txt")

    lines = []
    for dirpath, _, filenames in os.walk(split_root):
        for fname in filenames:
            if not fname.lower().endswith(".jpg"):
                continue

            img_path = Path(dirpath) / fname
            stem = img_path.stem
            txt_path = img_path.with_suffix(".txt")

            if not txt_path.is_file():
                print(f"[WARN] Missing txt for {img_path}")
                continue

            # relative path w.r.t split_root (what LMDB script expects)
            rel_img = img_path.relative_to(split_root)

            with open(txt_path, "r", encoding="utf-8") as f:
                label = f.read().strip()

            # create_lmdb_dataset.py uses split(maxsplit=1), so spaces in label are OK
            lines.append(f"{rel_img.as_posix()} {label}\n")

    print(f"{split}: {len(lines)} samples")
    with open(gt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {gt_path}")

if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        make_gt(s)

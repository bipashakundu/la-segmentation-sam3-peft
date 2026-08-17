"""
Verify that la_2018 GT masks match the COCO annotations from SAM3-LoRA.

Checks:
  1. Every img/ file has a matching gt/ file
  2. GT files are binary (only 0 and 255)
  3. Non-zero GT count matches COCO annotation count
  4. A random annotated slice: GT pixels overlap with COCO polygon bbox
  5. Saves a side-by-side overlay image for visual inspection
"""

import json
import os
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw

SRC_COCO = Path("./data_laseg2018")
DST      = Path("./load/la_2018")
OUT_VIS  = Path("./gt_verify_la2018.png")

SPLITS   = [("train", "train"), ("valid", "val"), ("test", "test")]

all_ok = True

for src_split, dst_split in SPLITS:
    print(f"\n── {dst_split} ──")
    img_dir = DST / dst_split / "img"
    gt_dir  = DST / dst_split / "gt"
    coco_path = SRC_COCO / src_split / "_annotations.coco.json"

    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    gt_files  = sorted(f for f in os.listdir(gt_dir)  if f.endswith(".jpg"))

    # 1. Matching files
    if img_files != gt_files:
        print(f"  FAIL: img/gt mismatch — {len(img_files)} imgs vs {len(gt_files)} gts")
        all_ok = False
    else:
        print(f"  OK  : {len(img_files)} img/gt pairs match")

    # 2. Binary check (sample 50 random gt files)
    sample = random.sample(gt_files, min(50, len(gt_files)))
    bad_binary = []
    for f in sample:
        arr = np.array(Image.open(gt_dir / f).convert("L"))
        unique = set(np.unique(arr).tolist())
        if not unique.issubset({0, 255}):
            bad_binary.append(f)
    if bad_binary:
        print(f"  FAIL: {len(bad_binary)} GT files have non-binary values: {bad_binary[:3]}")
        all_ok = False
    else:
        print(f"  OK  : GT files are binary (0/255) — checked {len(sample)} random files")

    # 3. Non-zero GT count vs COCO annotation count
    with open(coco_path) as f:
        coco = json.load(f)
    annotated_ids = {a["image_id"] for a in coco["annotations"]}
    annotated_fnames = set()
    for img_info in coco["images"]:
        if img_info["id"] in annotated_ids:
            annotated_fnames.add(img_info["file_name"].replace(".png", ".jpg"))

    gt_nonzero = sum(
        1 for f in gt_files
        if np.array(Image.open(gt_dir / f).convert("L")).max() > 0
    )
    print(f"  COCO annotated slices : {len(annotated_fnames)}")
    print(f"  GT files with mask    : {gt_nonzero}")
    if abs(gt_nonzero - len(annotated_fnames)) > 5:
        print(f"  WARN: count mismatch (diff={abs(gt_nonzero - len(annotated_fnames))})")
    else:
        print(f"  OK  : counts match")

    # 4. Spot-check: pick one annotated slice, check GT covers the COCO bbox
    ann_sample = random.choice(coco["annotations"])
    img_map = {i["id"]: i for i in coco["images"]}
    img_info = img_map[ann_sample["image_id"]]
    check_fname = img_info["file_name"].replace(".png", ".jpg")
    gt_arr = np.array(Image.open(gt_dir / check_fname).convert("L"))
    bx, by, bw, bh = [int(v) for v in ann_sample["bbox"]]
    roi_gt = gt_arr[by:by+bh, bx:bx+bw]
    coverage = roi_gt.max()
    if coverage == 0:
        print(f"  FAIL: spot-check — GT is empty inside COCO bbox for {check_fname}")
        all_ok = False
    else:
        print(f"  OK  : spot-check — GT has mask inside COCO bbox for {check_fname}")

    # 5. Visual overlay for train split only
    if dst_split == "train":
        img_arr = np.array(Image.open(img_dir / check_fname).convert("RGB"))
        overlay = img_arr.copy()
        overlay[gt_arr > 127] = [255, 80, 80]   # red = predicted mask
        blended = (0.6 * img_arr + 0.4 * overlay).astype(np.uint8)
        # Draw bbox
        cv2.rectangle(blended, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
        combined = np.concatenate([img_arr, blended], axis=1)
        Image.fromarray(combined).save(OUT_VIS)
        print(f"  Visual overlay saved → {OUT_VIS}")

print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")

#!/usr/bin/env python3
"""
Rebuild the SAM3 dataset from LAScarQS NIfTI files with proper patient-wise splits.

- la_001..la_104 (104 patients) → 84 train / 20 valid  (no patient appears in both)
- la_105..la_130 (26 patients)  → test                 (already held out by dataset)

Output:
  data_v2/train/_annotations.coco.json + images
  data_v2/valid/_annotations.coco.json + images
  data_v2/test/_annotations.coco.json  + images
"""

import os, json, random
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
IMAGES_TR   = Path("<PATH_TO_PROJECT_ROOT>/segmentation/nnUNet_raw/Dataset214_Heart/imagesTr")
LABELS_TR   = Path("<PATH_TO_PROJECT_ROOT>/segmentation/nnUNet_raw/Dataset214_Heart/labelsTr")
IMAGES_TS   = Path("<PATH_TO_PROJECT_ROOT>/segmentation/nnUNet_raw/Dataset214_Heart/imagesTs")
LABELS_TS   = Path("<PATH_TO_PROJECT_ROOT>/segmentation/nnUNet_raw/Dataset214_Heart/gt_test")
OUTPUT_DIR  = Path("./data_v2")

VAL_PATIENTS   = 20          # out of 104 training patients
SEED           = 42
MIN_MASK_PIXELS = 10          # ignore slices with fewer foreground pixels than this

CATEGORY = [{"id": 1, "name": "left atrium", "supercategory": "cardiac"}]


def normalize_slice(arr2d):
    """Normalize a 2D MRI slice to uint8 [0,255] using percentile clipping."""
    lo  = np.percentile(arr2d, 1)
    hi  = np.percentile(arr2d, 99)
    arr = np.clip(arr2d, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def mask_to_polygon(mask_2d):
    """Convert a binary 2D mask to a list of COCO polygon points."""
    contours, _ = cv2.findContours(
        mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for c in contours:
        if c.shape[0] < 3:
            continue
        poly = c.flatten().tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    return polygons


def process_patient(patient_id, img_path, lbl_path, out_dir,
                    image_id_start, ann_id_start):
    """
    Extract 2D slices from one patient, save images, build COCO entries.
    Returns (images_list, annotations_list, next_image_id, next_ann_id).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    img_nib = nib.load(img_path)
    lbl_nib = nib.load(lbl_path)

    img_data = img_nib.get_fdata()   # (H, W, D)
    lbl_data = lbl_nib.get_fdata()   # (H, W, D)

    n_slices = img_data.shape[2]
    images, annotations = [], []
    iid, aid = image_id_start, ann_id_start

    for s in range(n_slices):
        slice_2d = img_data[:, :, s]
        mask_2d  = (lbl_data[:, :, s] > 0.5).astype(np.uint8)

        # Normalize and save as RGB PNG
        norm = normalize_slice(slice_2d)
        rgb  = np.stack([norm, norm, norm], axis=-1)
        fname = f"{patient_id}_s{s:03d}.png"
        PILImage.fromarray(rgb).save(out_dir / fname)

        h, w = slice_2d.shape
        images.append({"id": iid, "file_name": fname, "width": w, "height": h})

        # Annotation only if enough foreground pixels
        if mask_2d.sum() >= MIN_MASK_PIXELS:
            polygons = mask_to_polygon(mask_2d)
            if polygons:
                xs = [p for poly in polygons for p in poly[0::2]]
                ys = [p for poly in polygons for p in poly[1::2]]
                x1, y1 = min(xs), min(ys)
                bw, bh = max(xs) - x1, max(ys) - y1
                area    = int(mask_2d.sum())
                annotations.append({
                    "id": aid,
                    "image_id": iid,
                    "category_id": 1,
                    "segmentation": polygons,
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "area": area,
                    "iscrowd": 0
                })
                aid += 1

        iid += 1

    return images, annotations, iid, aid


def build_coco(images, annotations):
    return {
        "info": {"description": "LAScarQS 2022 Task2 - patient-wise split"},
        "images": images,
        "annotations": annotations,
        "categories": CATEGORY
    }


def main():
    random.seed(SEED)

    # ── Collect training patients ──────────────────────────────────────────────
    tr_patients = sorted([
        f.name.replace("_0000.nii.gz", "")
        for f in IMAGES_TR.glob("*.nii.gz")
    ])
    ts_patients = sorted([
        f.name.replace("_0000.nii.gz", "")
        for f in IMAGES_TS.glob("*.nii.gz")
    ])

    print(f"Training patients : {len(tr_patients)}")
    print(f"Test patients     : {len(ts_patients)}")

    # Patient-wise train/val split
    shuffled = tr_patients.copy()
    random.shuffle(shuffled)
    val_patients   = set(shuffled[:VAL_PATIENTS])
    train_patients = set(shuffled[VAL_PATIENTS:])

    print(f"Split → train={len(train_patients)}, valid={len(val_patients)}, test={len(ts_patients)}")

    splits = {
        "train": (train_patients, IMAGES_TR, LABELS_TR),
        "valid": (val_patients,   IMAGES_TR, LABELS_TR),
        "test":  (set(ts_patients), IMAGES_TS, LABELS_TS),
    }

    for split_name, (patient_set, img_dir, lbl_dir) in splits.items():
        out_dir = OUTPUT_DIR / split_name
        # Skip if already completed
        if (out_dir / "_annotations.coco.json").exists():
            print(f"\n── Skipping {split_name} (already done) ──")
            continue
        print(f"\n── Building {split_name} ({len(patient_set)} patients) ──")
        all_images, all_anns = [], []
        iid, aid = 0, 0

        for pid in tqdm(sorted(patient_set), desc=split_name):
            img_path = img_dir / f"{pid}_0000.nii.gz"
            lbl_path = lbl_dir / f"{pid}.nii.gz"

            if not img_path.exists():
                print(f"  WARNING: missing image {img_path}")
                continue
            if not lbl_path.exists():
                print(f"  WARNING: missing label {lbl_path}")
                continue

            imgs, anns, iid, aid = process_patient(
                pid, img_path, lbl_path, out_dir, iid, aid
            )
            all_images.extend(imgs)
            all_anns.extend(anns)

        coco = build_coco(all_images, all_anns)
        ann_file = out_dir / "_annotations.coco.json"
        with open(ann_file, "w") as f:
            json.dump(coco, f)

        annotated = len({a["image_id"] for a in all_anns})
        empty     = len(all_images) - annotated
        print(f"  Saved {len(all_images)} images ({annotated} annotated, {empty} empty slices)")
        print(f"  Annotations : {len(all_anns)}")
        print(f"  COCO file   : {ann_file}")

    print("\n✅ Done. Update data_dir in configs/full_lora_config.yaml to use data_v2/")


if __name__ == "__main__":
    main()

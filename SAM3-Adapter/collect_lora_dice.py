"""
Collect per-slice Dice from SAM3-LoRA on annotated test slices only.
Saves a dict {filename: dice} as a .npy file for statistical testing.

Usage:
  python collect_lora_dice.py --dataset 2022
  python collect_lora_dice.py --dataset 2018
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image as PILImage

sys.path.insert(0, ".")

from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
import pycocotools.mask as mask_utils
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "2022": {
        "data_dir":   "./data_v2/test",
        "weights":    "./outputs/run_20260602_190336/best_lora_weights.pt",
        "out":        "./results/lora_2022_per_slice_dice.npy",
        "img_size":   1008,
    },
    "2018": {
        "data_dir":   "./data_laseg2018/test",
        "weights":    "./outputs/laseg2018_run_20260602_190336/best_lora_weights.pt",
        "out":        "./results/lora_2018_per_slice_dice.npy",
        "img_size":   1008,
    },
}

LORA_CFG = LoRAConfig(rank=8, alpha=16.0, dropout=0.0,
                       target_modules=["qkv", "proj", "fc1", "fc2"])


def poly_to_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def rle_to_mask(rle, h, w):
    if isinstance(rle, dict):
        return mask_utils.decode(rle).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def dice_score(pred_bin, gt_bin):
    inter = np.sum(pred_bin * gt_bin)
    denom = np.sum(pred_bin) + np.sum(gt_bin)
    if denom == 0:
        return 1.0
    return float(2 * inter / (denom + 1e-6))


def preprocess_image(img_path, img_size):
    img = PILImage.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size), PILImage.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


def build_model(weights_path):
    model = build_sam3_image_model(model_type="large")
    apply_lora_to_model(model, LORA_CFG)
    load_lora_weights(model, weights_path)
    model = model.to(device).eval()
    return model


def run_inference(model, img_tensor, img_size):
    """Run SAM3-LoRA and return binary mask (img_size x img_size)."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        try:
            output = model(img_tensor)
            # Extract first mask prediction
            if hasattr(output, "pred_masks"):
                masks = output.pred_masks
            elif isinstance(output, dict) and "pred_masks" in output:
                masks = output["pred_masks"]
            else:
                return np.zeros((img_size, img_size), dtype=np.uint8)

            if masks is None or masks.numel() == 0:
                return np.zeros((img_size, img_size), dtype=np.uint8)

            # Take highest-scoring mask
            mask = masks[0]
            if mask.ndim == 4:
                mask = mask[0, 0]
            elif mask.ndim == 3:
                mask = mask[0]
            pred_bin = (torch.sigmoid(mask) > 0.5).cpu().numpy().astype(np.uint8)
            return pred_bin
        except Exception:
            return np.zeros((img_size, img_size), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["2022", "2018"], required=True)
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)

    ann_file = os.path.join(cfg["data_dir"], "_annotations.coco.json")
    with open(ann_file) as f:
        coco = json.load(f)

    # Build image_id → annotations index
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Sort images by filename for consistent ordering
    images = sorted(coco["images"], key=lambda x: x["file_name"])

    model = build_model(cfg["weights"])
    print(f"Model loaded: {cfg['weights']}")

    all_names = []
    all_dice  = []

    for img_info in tqdm(images, desc=f"LoRA {args.dataset}"):
        iid   = img_info["id"]
        fname = img_info["file_name"]
        h, w  = img_info["height"], img_info["width"]

        # Only process annotated slices
        if iid not in ann_by_img:
            continue

        img_path = os.path.join(cfg["data_dir"], fname)
        if not os.path.exists(img_path):
            continue

        # Build GT mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in ann_by_img[iid]:
            seg = ann["segmentation"]
            if isinstance(seg, list):
                gt_mask = np.maximum(gt_mask, poly_to_mask(seg, h, w))
            elif isinstance(seg, dict):
                gt_mask = np.maximum(gt_mask, rle_to_mask(seg, h, w))

        if gt_mask.max() == 0:
            continue  # skip unannotated

        # Run model
        img_tensor = preprocess_image(img_path, cfg["img_size"])
        pred_bin   = run_inference(model, img_tensor, cfg["img_size"])

        # Resize GT to model output size for comparison
        gt_resized = cv2.resize(gt_mask, (cfg["img_size"], cfg["img_size"]),
                                interpolation=cv2.INTER_NEAREST)

        d = dice_score(pred_bin, gt_resized)
        all_names.append(fname)
        all_dice.append(d)

    result = {"filenames": all_names, "dice": np.array(all_dice, dtype=np.float32)}
    np.save(cfg["out"], result, allow_pickle=True)

    print(f"\nAnnotated slices: {len(all_dice)}")
    print(f"Mean Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Saved to: {cfg['out']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run inference on test images, save predicted overlays, and compute per-image IoU and Dice.

Output directory structure:
  <output_dir>/
    overlays/       <- side-by-side: original | GT | prediction
    pred_masks/     <- binary predicted masks (white on black)
    metrics.csv     <- per-image IoU and Dice
"""

import os, argparse, json, csv
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
import pycocotools.mask as mask_utils
import cv2


# ─── Dataset (reuses validate_sam3_lora approach) ────────────────────────────

class TestDataset(Dataset):
    def __init__(self, data_dir, query_text_map=None, filter_image_ids=None):
        self.data_dir = Path(data_dir)
        ann_file = self.data_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
        with open(ann_file) as f:
            self.coco_data = json.load(f)

        self.images      = {img['id']: img for img in self.coco_data['images']}
        all_ids          = sorted(self.images.keys())
        self.image_ids   = [i for i in all_ids if i in set(filter_image_ids)] \
                           if filter_image_ids is not None else all_ids
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)
        self.categories  = {c['id']: c['name'] for c in self.coco_data['categories']}
        self.query_text_map = query_text_map or {}

        self.resolution  = 1008
        self.transform   = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.data_dir / img_info['file_name']

        pil_image        = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h   = pil_image.size
        resized          = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        tensor           = self.transform(resized)

        anns = self.img_to_anns.get(img_id, [])
        objects = []
        for ann in anns:
            # Build binary GT mask from polygon segmentation
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if 'segmentation' in ann and ann['segmentation']:
                seg = ann['segmentation']
                if isinstance(seg, list):
                    for poly in seg:
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 1)
                elif isinstance(seg, dict):  # RLE
                    mask = mask_utils.decode(seg).astype(np.uint8)

            mask_resized = cv2.resize(mask, (self.resolution, self.resolution),
                                      interpolation=cv2.INTER_NEAREST)
            mask_tensor  = torch.from_numpy(mask_resized).bool()

            x, y, bw, bh = ann['bbox']
            box_xyxy = torch.tensor([x / orig_w, y / orig_h,
                                     (x + bw) / orig_w, (y + bh) / orig_h],
                                    dtype=torch.float32)

            area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
            objects.append(Object(
                segment=mask_tensor,
                bbox=box_xyxy,
                area=float(area),
                object_id=len(objects),
            ))

        cat_name   = self.categories.get(1, "left atrium")
        query_text = self.query_text_map.get(cat_name, cat_name)
        find_query = FindQueryLoaded(
            query_text=query_text,
            image_id=0,
            object_ids_output=[o.object_id for o in objects],
            is_exhaustive=True,
            inference_metadata=InferenceMetadata(
                coco_image_id=img_id,
                original_image_id=img_id,
                original_category_id=0,
                original_size=(orig_h, orig_w),
                object_id=-1,
                frame_index=-1,
            ),
        )

        return Datapoint(
            images=[Image(data=tensor, objects=objects, size=(self.resolution, self.resolution))],
            find_queries=[find_query],
        )


# ─── Utilities ────────────────────────────────────────────────────────────────

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, move_to_device(getattr(obj, field), device))
        return obj
    return obj


def get_best_mask(pred_logits, pred_masks, orig_h, orig_w, prob_threshold=0.3):
    """
    Return the single best binary mask at original resolution.
    Takes the highest-scoring prediction above threshold — correct for
    single-object segmentation (left atrium). Using a union of all masks
    above threshold includes false positives and destroys IoU/Dice.
    """
    scores = torch.sigmoid(pred_logits).squeeze(-1)   # [N]
    keep   = scores > prob_threshold

    if keep.sum() == 0:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    masks_keep  = pred_masks[keep]    # [K, 288, 288]
    scores_keep = scores[keep]        # [K]

    # Take only the single highest-scoring mask
    best_idx  = scores_keep.argmax()
    best_mask = masks_keep[best_idx]  # [288, 288]

    mask_sigmoid = torch.sigmoid(best_mask)
    mask_up = torch.nn.functional.interpolate(
        mask_sigmoid.unsqueeze(0).unsqueeze(0).float(),
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze()                                        # [H, W]

    return (mask_up > 0.5).cpu().numpy().astype(np.uint8)


def compute_gt_mask(dataset, idx, orig_h, orig_w):
    """Build the merged GT binary mask for one sample from its annotations."""
    img_id = dataset.image_ids[idx]
    anns   = dataset.img_to_anns.get(img_id, [])
    gt     = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for ann in anns:
        seg = ann.get('segmentation', [])
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(gt, [pts], 1)
        elif isinstance(seg, dict):
            gt = np.maximum(gt, mask_utils.decode(seg).astype(np.uint8))
    return gt


def iou_dice(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou  = inter / (union + 1e-7)
    dice = 2 * inter / (pred.sum() + gt.sum() + 1e-7)
    return float(iou), float(dice)


def save_overlay(orig_pil, gt_mask, pred_mask, out_path):
    """Save a three-panel image: original | GT overlay | prediction overlay."""
    orig = np.array(orig_pil.convert("RGB"))
    h, w = orig.shape[:2]

    def blend(base, mask, colour):
        out = base.copy()
        out[mask == 1] = (out[mask == 1] * 0.5 + np.array(colour) * 0.5).astype(np.uint8)
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, 2)
        return out

    gt_panel   = blend(orig, gt_mask,   (0, 220, 0))    # green
    pred_panel = blend(orig, pred_mask, (220, 50, 50))   # red

    canvas = np.concatenate([orig, gt_panel, pred_panel], axis=1)
    PILImage.fromarray(canvas).save(out_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(config_path, weights_path, test_data_dir, output_dir,
        prob_threshold=0.3, use_base_model=False, filter_image_ids=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = build_sam3_image_model(
        device=device.type, compile=False, load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
    )

    if not use_base_model:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        lc = config["lora"]
        lora_config = LoRAConfig(
            rank=lc["rank"], alpha=lc["alpha"], dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            apply_to_vision_encoder=lc["apply_to_vision_encoder"],
            apply_to_text_encoder=lc["apply_to_text_encoder"],
            apply_to_geometry_encoder=lc["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lc["apply_to_detr_encoder"],
            apply_to_detr_decoder=lc["apply_to_detr_decoder"],
            apply_to_mask_decoder=lc["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)
        print(f"Loading LoRA weights from {weights_path}...")
        load_lora_weights(model, weights_path)

    model.to(device).eval()

    # Prepare output dirs
    out_dir      = Path(output_dir)
    overlay_dir  = out_dir / "overlays"
    pred_dir     = out_dir / "pred_masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Dataset — pass query_text_map from config so prompt matches training
    query_text_map = {}
    if not use_base_model and config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        query_text_map = cfg.get("training", {}).get("query_text_map", {})
    dataset = TestDataset(test_data_dir, query_text_map=query_text_map,
                          filter_image_ids=filter_image_ids)
    print(f"Loaded {len(dataset)} test images from {test_data_dir}")
    print(f"Query text map: {query_text_map}")

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    rows = []
    sample_idx = 0

    with torch.no_grad():
        for batch_dict in tqdm(loader, desc="Inference"):
            input_batch = move_to_device(batch_dict["input"], device)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs_list = model(input_batch)
            else:
                outputs_list = model(input_batch)

            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_outputs = list(list(outputs_iter)[-1])[-1]

            actual_batch = final_outputs['pred_logits'].shape[0]

            for i in range(actual_batch):
                pred_logits = final_outputs['pred_logits'][i].detach().cpu()   # [N,1]
                pred_masks  = final_outputs['pred_masks'][i].detach().cpu()    # [N,288,288]

                img_id   = dataset.image_ids[sample_idx]
                img_info = dataset.images[img_id]
                fname    = img_info['file_name']
                orig_pil = PILImage.open(dataset.data_dir / fname).convert("RGB")
                orig_w, orig_h = orig_pil.size

                pred_mask = get_best_mask(pred_logits, pred_masks, orig_h, orig_w, prob_threshold)
                gt_mask   = compute_gt_mask(dataset, sample_idx, orig_h, orig_w)

                iou, dice = iou_dice(pred_mask, gt_mask)
                has_gt = int(gt_mask.sum() > 0)
                rows.append({'file': fname, 'iou': f'{iou:.4f}', 'dice': f'{dice:.4f}', 'has_gt': has_gt})

                stem = Path(fname).stem
                save_overlay(orig_pil, gt_mask, pred_mask, overlay_dir / f"{stem}_overlay.png")

                pred_img = PILImage.fromarray((pred_mask * 255).astype(np.uint8))
                pred_img.save(pred_dir / f"{stem}_pred.png")

                sample_idx += 1

    # Save CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'iou', 'dice', 'has_gt'])
        writer.writeheader()
        writer.writerows(rows)

    # Split into annotated vs empty slices
    ann_rows   = [r for r in rows if int(r['has_gt']) == 1]
    empty_rows = [r for r in rows if int(r['has_gt']) == 0]

    all_ious   = [float(r['iou'])  for r in rows]
    all_dices  = [float(r['dice']) for r in rows]
    ann_ious   = [float(r['iou'])  for r in ann_rows]
    ann_dices  = [float(r['dice']) for r in ann_rows]

    print(f"\n{'='*60}")
    print(f"Results: {len(rows)} total slices  "
          f"({len(ann_rows)} annotated, {len(empty_rows)} empty)")
    print(f"{'='*60}")
    print(f"  [Annotated slices only — use this for paper]")
    print(f"  Mean IoU  : {np.mean(ann_ious):.4f}  (std {np.std(ann_ious):.4f})")
    print(f"  Mean Dice : {np.mean(ann_dices):.4f}  (std {np.std(ann_dices):.4f})")
    print(f"")
    print(f"  [All slices — includes empty slices penalising false positives]")
    print(f"  Mean IoU  : {np.mean(all_ious):.4f}  (std {np.std(all_ious):.4f})")
    print(f"  Mean Dice : {np.mean(all_dices):.4f}  (std {np.std(all_dices):.4f})")
    print(f"{'='*60}")
    print(f"Overlays saved to : {overlay_dir}")
    print(f"Pred masks saved to: {pred_dir}")
    print(f"Per-image CSV      : {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save predicted overlays and compute per-image IoU / Dice"
    )
    parser.add_argument("--config",  type=str, default=None,
                        help="Path to training config YAML (for LoRA settings)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to LoRA weights .pt file")
    parser.add_argument("--test_data_dir", type=str, required=True,
                        help="Directory with images and _annotations.coco.json")
    parser.add_argument("--output_dir",    type=str, default="outputs/visualizations",
                        help="Where to save overlays and CSV (default: outputs/visualizations)")
    parser.add_argument("--prob_threshold", type=float, default=0.3,
                        help="Score threshold for keeping predictions (default: 0.3)")
    parser.add_argument("--use_base_model", action="store_true",
                        help="Use base SAM3 without LoRA (for baseline comparison)")
    parser.add_argument("--image_ids", type=int, nargs="+", default=None,
                        help="Only visualise these COCO image IDs (e.g. --image_ids 986 693 100)")
    args = parser.parse_args()

    if not args.use_base_model and (args.config is None or args.weights is None):
        parser.error("--config and --weights are required unless --use_base_model is set")

    run(
        config_path=args.config,
        weights_path=args.weights,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        prob_threshold=args.prob_threshold,
        use_base_model=args.use_base_model,
        filter_image_ids=args.image_ids,
    )

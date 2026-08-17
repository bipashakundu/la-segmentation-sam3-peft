"""
Collect per-slice Dice from SAM3-Adapter on annotated test slices only.
Saves a dict {filename: dice} as a .npy file for statistical testing.

Usage:
  python collect_adapter_dice.py --dataset 2022
  python collect_adapter_dice.py --dataset 2018
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import datasets
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "2022": {
        "config":   "configs/left-atrium-sam-vit-l-v2.yaml",
        "checkpoint": "save/_left-atrium-sam-vit-l-v2/model_epoch_best.pth",
        "out":      "results/adapter_2022_per_slice_dice.npy",
    },
    "2018": {
        "config":   "configs/laseg2018-sam-vit-l.yaml",
        "checkpoint": "save/_laseg2018-sam-vit-l/model_epoch_best.pth",
        "out":      "results/adapter_2018_per_slice_dice.npy",
    },
}


def dice_per_slice(pred_prob, gt, threshold=0.5):
    """Compute Dice for each sample in a batch. Returns numpy array of shape (B,)."""
    pred_bin = (pred_prob.detach().cpu().numpy() > threshold).astype(np.float32)
    gt_bin   = (gt.detach().cpu().numpy()        > 0.5).astype(np.float32)
    dims = tuple(range(1, pred_bin.ndim))
    inter  = np.sum(pred_bin * gt_bin, axis=dims)
    sum_p  = np.sum(pred_bin,          axis=dims)
    sum_t  = np.sum(gt_bin,            axis=dims)
    denom  = sum_p + sum_t
    dice   = np.where(denom == 0, 1.0, 2 * inter / (denom + 1e-6))
    return dice  # shape (B,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["2022", "2018"], required=True)
    args = parser.parse_args()

    cfg_info = CONFIGS[args.dataset]
    os.makedirs("results", exist_ok=True)

    with open(cfg_info["config"]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build test dataset + loader
    spec    = config["test_dataset"]
    dataset = datasets.make(spec["dataset"])

    # Grab sorted filenames from the GT folder (dataset_2 = GT)
    raw_gt_dataset = dataset.dataset_2
    filenames = [os.path.basename(p) for p in raw_gt_dataset.files]

    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    loader  = DataLoader(dataset, batch_size=spec["batch_size"],
                         num_workers=4, shuffle=False)

    # Load model
    model = models.make(config["model"]).cuda()
    ckpt  = torch.load(cfg_info["checkpoint"], map_location="cuda")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {cfg_info['checkpoint']}")

    bs        = spec["batch_size"]
    all_names = []
    all_dice  = []

    idx = 0
    for batch in tqdm(loader, desc=f"Adapter {args.dataset}"):
        inp = batch["inp"].cuda()
        gt  = batch["gt"].cuda()

        with torch.no_grad():
            pred = torch.sigmoid(model.infer(inp))

        dices    = dice_per_slice(pred, gt)   # (B,)
        gt_np    = gt.detach().cpu().numpy()  # (B,1,H,W)

        for i in range(len(dices)):
            global_idx = idx + i
            fname = filenames[global_idx] if global_idx < len(filenames) else f"slice_{global_idx}"
            gt_slice = gt_np[i]
            is_annotated = gt_slice.max() > 0.5

            if is_annotated:
                all_names.append(fname)
                all_dice.append(float(dices[i]))

        idx += len(dices)

    result = {"filenames": all_names, "dice": np.array(all_dice, dtype=np.float32)}
    np.save(cfg_info["out"], result, allow_pickle=True)

    print(f"\nAnnotated slices: {len(all_dice)}")
    print(f"Mean Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Saved to: {cfg_info['out']}")


if __name__ == "__main__":
    main()

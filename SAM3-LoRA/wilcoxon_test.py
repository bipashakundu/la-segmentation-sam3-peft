#!/usr/bin/env python3
"""
Wilcoxon signed-rank test for statistical significance between two segmentation methods.

Usage:
    python wilcoxon_test.py \
        --a  failure_analysis_2022_zeroshot/per_sample_metrics.csv \
        --b  failure_analysis_2022_r8/per_sample_metrics.csv \
        --label-a "Zero-shot SAM3" \
        --label-b "LoRA r=8" \
        --dataset "LAScarQS 2022"

Output: p-values, effect sizes, and a formatted results paragraph for the paper.
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def effect_size_r(statistic, n):
    """r = Z / sqrt(N), where Z is derived from the Wilcoxon statistic."""
    # scipy returns the test statistic W; compute Z approximation
    mean_w  = n * (n + 1) / 4
    std_w   = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z       = (statistic - mean_w) / std_w
    return abs(z) / np.sqrt(n)

def interpret_effect(r):
    if r < 0.1:   return "negligible"
    elif r < 0.3: return "small"
    elif r < 0.5: return "medium"
    else:         return "large"

def run_test(a_vals, b_vals, metric_name, subset_name, label_a, label_b):
    diffs = b_vals - a_vals
    n_nonzero = (diffs != 0).sum()

    if n_nonzero < 10:
        print(f"  [{subset_name} | {metric_name}] Too few non-zero differences ({n_nonzero}) — skipping")
        return None

    stat, p = wilcoxon(a_vals, b_vals, alternative='two-sided')
    r = effect_size_r(stat, n_nonzero)
    effect = interpret_effect(r)

    mean_a, std_a = a_vals.mean(), a_vals.std()
    mean_b, std_b = b_vals.mean(), b_vals.std()
    direction = "↑" if mean_b > mean_a else "↓"

    print(f"\n  [{subset_name} | {metric_name}]")
    print(f"    {label_a:30s}: {mean_a:.4f} ± {std_a:.4f}")
    print(f"    {label_b:30s}: {mean_b:.4f} ± {std_b:.4f}  {direction}")
    print(f"    W={stat:.1f}, p={p:.4e}, r={r:.3f} ({effect} effect)")
    print(f"    Significant (p<0.05): {'YES' if p < 0.05 else 'NO'}")

    return {
        'subset': subset_name, 'metric': metric_name,
        'mean_a': mean_a, 'std_a': std_a,
        'mean_b': mean_b, 'std_b': std_b,
        'W': stat, 'p': p, 'r': r, 'effect': effect,
        'significant': p < 0.05, 'n': n_nonzero,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a',         required=True, help='CSV for method A (baseline)')
    parser.add_argument('--b',         required=True, help='CSV for method B (comparison)')
    parser.add_argument('--label-a',   default='Method A')
    parser.add_argument('--label-b',   default='Method B')
    parser.add_argument('--dataset',   default='Dataset')
    parser.add_argument('--annotated-only', action='store_true',
                        help='Only test on annotated slices (has_gt=1)')
    args = parser.parse_args()

    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)

    # join on slice name
    merged = pd.merge(df_a, df_b, on='name', suffixes=('_a', '_b'))
    print(f"\nDataset     : {args.dataset}")
    print(f"Method A    : {args.label_a}  ({args.a})")
    print(f"Method B    : {args.label_b}  ({args.b})")
    print(f"Total pairs : {len(merged)}")

    results = []

    # ── All slices ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("="*65)

    for metric in ['iou', 'dice']:
        col_a = f'{metric}_a'
        col_b = f'{metric}_b'
        if col_a not in merged.columns:
            continue
        res = run_test(merged[col_a].values, merged[col_b].values,
                       metric.upper(), 'All slices', args.label_a, args.label_b)
        if res:
            results.append(res)

    # ── Annotated slices only ─────────────────────────────────────────────────
    ann = merged[merged['has_gt_a'] == 1]
    print(f"\n  (Annotated slices only: n={len(ann)})")
    for metric in ['iou', 'dice']:
        col_a = f'{metric}_a'
        col_b = f'{metric}_b'
        if col_a not in ann.columns:
            continue
        res = run_test(ann[col_a].values, ann[col_b].values,
                       metric.upper(), 'Annotated only', args.label_a, args.label_b)
        if res:
            results.append(res)

    # ── Paper-ready summary ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PAPER-READY SUMMARY")
    print("="*65)

    ann_iou  = next((r for r in results if r['subset']=='Annotated only' and r['metric']=='IOU'),  None)
    ann_dice = next((r for r in results if r['subset']=='Annotated only' and r['metric']=='DICE'), None)
    all_dice = next((r for r in results if r['subset']=='All slices'     and r['metric']=='DICE'), None)

    if ann_dice:
        sig_str = "statistically significant" if ann_dice['significant'] else "not statistically significant"
        print(f"""
The difference in mean Dice between {args.label_a} ({ann_dice['mean_a']:.4f} ± {ann_dice['std_a']:.4f}) \
and {args.label_b} ({ann_dice['mean_b']:.4f} ± {ann_dice['std_b']:.4f}) \
on annotated slices was {sig_str} \
(Wilcoxon signed-rank test, W={ann_dice['W']:.0f}, p={ann_dice['p']:.4e}, \
effect size r={ann_dice['r']:.3f} [{ann_dice['effect']}]).
""")

    if ann_iou:
        sig_str = "statistically significant" if ann_iou['significant'] else "not statistically significant"
        print(f"""
The IoU difference ({args.label_a}: {ann_iou['mean_a']:.4f} ± {ann_iou['std_a']:.4f} vs \
{args.label_b}: {ann_iou['mean_b']:.4f} ± {ann_iou['std_b']:.4f}) was {sig_str} \
(W={ann_iou['W']:.0f}, p={ann_iou['p']:.4e}, r={ann_iou['r']:.3f} [{ann_iou['effect']}]).
""")


if __name__ == '__main__':
    main()

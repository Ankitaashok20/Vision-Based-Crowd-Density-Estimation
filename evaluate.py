"""
evaluate.py — Evaluate a trained YoloX+CFE checkpoint on the PHD test set.

Usage:
    python evaluate.py --data_root /path/to/VOC2007 \
                       --checkpoint ./runs/best.pth  \
                       --save_dir   ./runs
"""

import os
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_loader
from model   import YoloXCFE


#  Helpers 

def box_iou_np(a, b):
    """Vectorised IoU: (N,4) × (M,4) → (N,M)."""
    ix1 = np.maximum(a[:, 0, None], b[None, :, 0])
    iy1 = np.maximum(a[:, 1, None], b[None, :, 1])
    ix2 = np.minimum(a[:, 2, None], b[None, :, 2])
    iy2 = np.minimum(a[:, 3, None], b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-7)


def nms_np(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-scores)
    keep  = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious  = box_iou_np(boxes[i:i+1], boxes[order[1:]]).flatten()
        order = order[1:][ious < iou_thresh]
    return np.array(keep, dtype=int)


@torch.no_grad()
def decode_preds(raw_outputs, img_size, conf_thresh=0.01):
    B   = raw_outputs[0].shape[0]
    dev = raw_outputs[0].device
    per_img = [[] for _ in range(B)]

    for out in raw_outputs:
        stride    = img_size // out.shape[-1]
        B_, C, H, W = out.shape
        gy, gx = torch.meshgrid(
            torch.arange(H, device=dev),
            torch.arange(W, device=dev),
            indexing='ij',
        )
        grid = torch.stack([gx, gy], -1).float().reshape(1, H * W, 2)
        o    = out.permute(0, 2, 3, 1).reshape(B_, H * W, C)

        xy    = (o[..., :2].sigmoid() + grid) * stride
        wh    = o[..., 2:4].exp().clamp(max=img_size) * stride
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], -1)

        obj_score = o[..., 4].sigmoid()
        cls_score = o[..., 5:].sigmoid().max(-1).values
        score     = obj_score * cls_score

        for b in range(B_):
            mask = score[b] > conf_thresh
            if mask.sum() == 0:
                continue
            per_img[b].append({
                'boxes':  boxes[b][mask].cpu().numpy(),
                'scores': score[b][mask].cpu().numpy(),
            })

    results = []
    for b in range(B):
        if not per_img[b]:
            results.append({'boxes': np.zeros((0, 4)), 'scores': np.zeros(0)})
            continue
        bx   = np.concatenate([d['boxes']  for d in per_img[b]])
        sc   = np.concatenate([d['scores'] for d in per_img[b]])
        keep = nms_np(bx, sc, iou_thresh=0.45)
        results.append({'boxes': bx[keep], 'scores': sc[keep]})
    return results


def compute_ap_at_iou(all_preds, all_gts, iou_thresh, small_only=False, small_px=32):
    tp_list, conf_list = [], []
    n_gt = 0

    for preds, gts in zip(all_preds, all_gts):
        pb, ps = preds['boxes'], preds['scores']
        gb     = gts['boxes']

        if small_only and len(gb):
            area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            gb   = gb[area < small_px ** 2]

        n_gt += len(gb)
        if len(pb) == 0:
            continue

        order      = np.argsort(-ps)
        pb, ps     = pb[order], ps[order]
        matched_gt = set()

        for box, score in zip(pb, ps):
            conf_list.append(score)
            if len(gb) == 0:
                tp_list.append(0)
                continue
            ious = box_iou_np(box[None], gb).flatten()
            best = int(ious.argmax())
            if ious[best] >= iou_thresh and best not in matched_gt:
                tp_list.append(1)
                matched_gt.add(best)
            else:
                tp_list.append(0)

    if not tp_list or n_gt == 0:
        return 0., np.array([0.]), np.array([0.]), n_gt

    order      = np.argsort(-np.array(conf_list))
    tp_arr     = np.array(tp_list)[order]
    cum_tp     = np.cumsum(tp_arr)
    cum_fp     = np.cumsum(1 - tp_arr)
    recalls    = cum_tp / (n_gt + 1e-7)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-7)
    ap = 0.
    for thr in np.linspace(0, 1, 101):
        mask = recalls >= thr
        ap  += precisions[mask].max() if mask.any() else 0.
    ap /= 101.
    return float(ap), recalls, precisions, n_gt


#  Main 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--save_dir',   type=str, default='./runs')
    p.add_argument('--model_size', type=str, default='s', choices=['s', 'm', 'l'])
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--img_size',   type=int, default=416)
    p.add_argument('--workers',    type=int, default=2)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model = YoloXCFE(args.model_size, nc=1).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]+1}  '
          f'(val_loss={ckpt["best_loss"]:.4f})')

    # Inference
    test_loader = get_loader(args.data_root, 'test',
                             args.batch_size, args.img_size, args.workers)
    all_preds, all_gts = [], []

    for imgs, targets in tqdm(test_loader, desc='Running inference'):
        imgs      = imgs.to(device)
        raw       = model(imgs)
        batch_preds = decode_preds(raw, args.img_size, conf_thresh=0.01)
        all_preds.extend(batch_preds)
        for tgt in targets:
            all_gts.append({'boxes': tgt['boxes'].numpy() * args.img_size})

    # mAP
    iou_thresholds = np.arange(0.5, 1.0, 0.05).round(2).tolist()
    ap_list, pr_curves = [], {}

    for iou_t in iou_thresholds:
        ap, rec, pre, _ = compute_ap_at_iou(all_preds, all_gts, iou_t)
        ap_list.append(ap)
        if abs(iou_t - 0.50) < 1e-4:
            pr_curves[0.50] = (rec, pre)
        if abs(iou_t - 0.75) < 1e-4:
            pr_curves[0.75] = (rec, pre)

    map50    = ap_list[0]
    map50_95 = float(np.mean(ap_list))

    ap_small_list = []
    for iou_t in iou_thresholds:
        ap, _, _, _ = compute_ap_at_iou(all_preds, all_gts, iou_t,
                                        small_only=True, small_px=32)
        ap_small_list.append(ap)
    maps50_95 = float(np.mean(ap_small_list))

    # Precision / Recall / F1 @ conf=0.5, IoU=0.5
    tp = fp = fn = 0
    per_img_tp, per_img_fp, per_img_fn = [], [], []

    for preds, gts in zip(all_preds, all_gts):
        pb, ps = preds['boxes'], preds['scores']
        gb     = gts['boxes']
        pb_c   = pb[ps >= 0.5]
        matched = set()
        img_tp = img_fp = 0

        for box in pb_c:
            if len(gb) == 0:
                img_fp += 1
                continue
            ious = box_iou_np(box[None], gb).flatten()
            best = int(ious.argmax())
            if ious[best] >= 0.5 and best not in matched:
                img_tp += 1
                matched.add(best)
            else:
                img_fp += 1

        img_fn = len(gb) - len(matched)
        tp += img_tp; fp += img_fp; fn += img_fn
        per_img_tp.append(img_tp)
        per_img_fp.append(img_fp)
        per_img_fn.append(img_fn)

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)

    # FPS
    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    for _ in range(5):
        with torch.no_grad():
            model(dummy)
    N_RUNS = 100
    t0 = time.time()
    for _ in range(N_RUNS):
        with torch.no_grad():
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fps = N_RUNS / (time.time() - t0)

    # Print
    print()
    print('=' * 56)
    print('  Evaluation Results  — YoloX(CFE) on PHD test set')
    print('=' * 56)
    print(f'  Precision  (P)         : {precision:.4f}')
    print(f'  Recall     (R)         : {recall:.4f}')
    print(f'  F1-score   (F1)        : {f1:.4f}')
    print(f'  mAP@0.5                : {map50:.4f}')
    print(f'  mAP@[0.5:0.95]         : {map50_95:.4f}')
    print(f'  mAPs@[0.5:0.95] (small): {maps50_95:.4f}')
    print(f'  FPS                    : {fps:.1f}')
    print('-' * 56)
    print(f'  TP / FP / FN           : {tp} / {fp} / {fn}')
    print(f'  Total GT boxes         : {sum(len(g["boxes"]) for g in all_gts)}')
    print(f'  Images evaluated       : {len(all_gts)}')
    print('=' * 56)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('YoloX(CFE) — PHD Test Set', fontsize=13, fontweight='bold')

    ax = axes[0]
    colors = {0.50: '#2563eb', 0.75: '#16a34a'}
    labels = {0.50: f'IoU=0.50  AP={map50:.3f}',
              0.75: f'IoU=0.75  AP={ap_list[5]:.3f}'}
    for t, (r, pr) in pr_curves.items():
        ax.plot(r, pr, color=colors[t], lw=1.8, label=labels[t])
    if 0.50 in pr_curves:
        ax.fill_between(pr_curves[0.50][0], pr_curves[0.50][1], alpha=0.08, color='#2563eb')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('PR Curve'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    ax = axes[1]
    ax.bar(iou_thresholds, ap_list, width=0.04, color='#7c3aed', alpha=0.75)
    ax.axhline(map50_95, color='#dc2626', lw=1.5, ls='--',
               label=f'mAP@[.5:.95]={map50_95:.3f}')
    ax.set_xlabel('IoU threshold'); ax.set_ylabel('AP')
    ax.set_title('AP vs IoU Threshold')
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(alpha=0.2, axis='y')

    ax    = axes[2]
    means = [np.mean(per_img_tp), np.mean(per_img_fp), np.mean(per_img_fn)]
    stds  = [np.std(per_img_tp),  np.std(per_img_fp),  np.std(per_img_fn)]
    clrs  = ['#16a34a', '#dc2626', '#d97706']
    bars  = ax.bar([0, 1, 2], means, yerr=stds, capsize=5,
                   color=clrs, alpha=0.8, width=0.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['TP', 'FP', 'FN'])
    ax.set_ylabel('Count per image (mean ± std)')
    ax.set_title('Per-image Detection Stats')
    ax.grid(alpha=0.2, axis='y')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{m:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, 'eval_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Plots saved → {plot_path}')


if __name__ == '__main__':
    main()
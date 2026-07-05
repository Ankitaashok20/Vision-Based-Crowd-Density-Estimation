"""
visualize.py — Visualise model predictions vs ground truth on test images.

Usage:
    python visualize.py --data_root /path/to/VOC2007 \
                        --checkpoint ./runs/best.pth  \
                        --save_dir   ./runs           \
                        --n_images   5
"""

import os
import argparse
import random

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset  import PHDDataset
from model    import YoloXCFE
from evaluate import decode_preds        # reuse decode helper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   type=str, required=True)
    p.add_argument('--checkpoint',  type=str, required=True)
    p.add_argument('--save_dir',    type=str, default='./runs')
    p.add_argument('--model_size',  type=str, default='s', choices=['s', 'm', 'l'])
    p.add_argument('--img_size',    type=int, default=416)
    p.add_argument('--n_images',    type=int, default=5)
    p.add_argument('--conf_thresh', type=float, default=0.5)
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

    # Dataset
    test_ds = PHDDataset(args.data_root, split='test', img_size=args.img_size)
    indices = random.sample(range(len(test_ds)), min(args.n_images, len(test_ds)))

    fig, axes = plt.subplots(args.n_images, 2, figsize=(14, args.n_images * 4))
    if args.n_images == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Actual vs Predicted Passenger Heads',
                 fontsize=15, fontweight='bold', y=1.01)

    for row, idx in enumerate(indices):
        img_tensor, target = test_ds[idx]
        img_id = target['image_id']

        # Load original resolution image
        img_path = os.path.join(args.data_root, 'JPEGImages', f'{img_id}.jpg')
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        disp_h, disp_w = orig_img.shape[:2]

        # Ground truth
        gt_norm  = target['boxes'].numpy()
        gt_orig  = gt_norm.copy()
        gt_orig[:, [0, 2]] *= disp_w
        gt_orig[:, [1, 3]] *= disp_h
        n_actual = len(gt_orig)

        # Predict
        inp = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(inp)
        preds  = decode_preds(raw, args.img_size, conf_thresh=0.01)
        pb_all = preds[0]['boxes']
        ps_all = preds[0]['scores']

        mask    = ps_all >= args.conf_thresh
        pb_conf = pb_all[mask]
        ps_conf = ps_all[mask]

        # Scale to original size
        sx = disp_w / args.img_size
        sy = disp_h / args.img_size
        pb_orig = pb_conf.astype(float).copy()
        if len(pb_orig):
            pb_orig[:, 0] *= sx; pb_orig[:, 2] *= sx
            pb_orig[:, 1] *= sy; pb_orig[:, 3] *= sy

        n_predicted = len(pb_orig)

        # Ground-truth panel
        ax_gt = axes[row, 0]
        ax_gt.imshow(orig_img)
        for box in gt_orig:
            x1, y1, x2, y2 = box
            ax_gt.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='#00e676', facecolor='none'))
        ax_gt.set_title(f'ACTUAL — {n_actual} heads\n(Image: {img_id})',
                        fontsize=10, fontweight='bold', color='#00c853')
        ax_gt.axis('off')

        # Prediction panel
        ax_pr = axes[row, 1]
        ax_pr.imshow(orig_img)
        for box, score in zip(pb_orig, ps_conf):
            x1, y1, x2, y2 = box
            ax_pr.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='#ff1744', facecolor='none'))
            ax_pr.text(x1, y1 - 3, f'{score:.2f}', fontsize=6, color='#ff1744',
                       bbox=dict(facecolor='white', alpha=0.4, pad=1, linewidth=0))
        ax_pr.set_title(
            f'PREDICTED — {n_predicted} heads\n(conf ≥ {args.conf_thresh})',
            fontsize=10, fontweight='bold', color='#d50000')
        ax_pr.axis('off')

        diff = n_predicted - n_actual
        sign = '+' if diff >= 0 else ''
        print(f'[{row+1}] {img_id:20s}  actual={n_actual:3d}  '
              f'predicted={n_predicted:3d}  diff={sign}{diff}')

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 'head_count_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved → {save_path}')


if __name__ == '__main__':
    main()
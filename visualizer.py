"""
results_visualizer.py — Complete results visualization for YoloX+CFE PHD head detection.

Generates:
  1. ≥10 images with bounding boxes (GT vs Predicted), head counts shown
  2. Dataset overview (split distribution, annotation stats, sample images)
  3. Performance metrics table (P, R, F1, mAP@50, mAP@50:95)
  4. Confusion matrix
  5. ROC + AUC curve (objectness-based)
  6. PR curve
  7. AP vs IoU threshold bar chart
  8. Per-image TP/FP/FN stats

Usage:
    python results_visualizer.py \
        --data_root /path/to/VOC2007 \
        --checkpoint ./runs/best.pth \
        --save_dir   ./results \
        --n_images   10 \
        --conf_thresh 0.5
"""

import os
import argparse
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

#  Imports from your project 
from dataset import PHDDataset, get_loader
from model   import YoloXCFE



# Helpers (copied / adapted from evaluate.py so this file is self-contained)


def box_iou_np(a, b):
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

        # also keep raw obj score for ROC
        raw_obj = o[..., 4].sigmoid()

        for b in range(B_):
            mask = score[b] > conf_thresh
            if mask.sum() == 0:
                continue
            per_img[b].append({
                'boxes':   boxes[b][mask].cpu().numpy(),
                'scores':  score[b][mask].cpu().numpy(),
                'raw_obj': raw_obj[b][mask].cpu().numpy(),
            })

    results = []
    for b in range(B):
        if not per_img[b]:
            results.append({'boxes': np.zeros((0, 4)), 'scores': np.zeros(0), 'raw_obj': np.zeros(0)})
            continue
        bx   = np.concatenate([d['boxes']  for d in per_img[b]])
        sc   = np.concatenate([d['scores'] for d in per_img[b]])
        ro   = np.concatenate([d['raw_obj'] for d in per_img[b]])
        keep = nms_np(bx, sc, iou_thresh=0.45)
        results.append({'boxes': bx[keep], 'scores': sc[keep], 'raw_obj': ro[keep]})
    return results


def compute_ap_at_iou(all_preds, all_gts, iou_thresh):
    tp_list, conf_list = [], []
    n_gt = 0

    for preds, gts in zip(all_preds, all_gts):
        pb, ps = preds['boxes'], preds['scores']
        gb     = gts['boxes']
        n_gt  += len(gb)
        if len(pb) == 0:
            continue
        order      = np.argsort(-ps)
        pb, ps     = pb[order], ps[order]
        matched_gt = set()
        for box, score in zip(pb, ps):
            conf_list.append(score)
            if len(gb) == 0:
                tp_list.append(0); continue
            ious = box_iou_np(box[None], gb).flatten()
            best = int(ious.argmax())
            if ious[best] >= iou_thresh and best not in matched_gt:
                tp_list.append(1); matched_gt.add(best)
            else:
                tp_list.append(0)

    if not tp_list or n_gt == 0:
        return 0., np.array([0.]), np.array([0.])

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
    return float(ap), recalls, precisions


# 1. BOUNDING BOX VISUALISATION  (≥10 images)

def plot_bbox_grid(test_ds, model, device, args):
    n = max(10, args.n_images)
    indices = random.sample(range(len(test_ds)), min(n, len(test_ds)))

    cols  = 2          # GT | Pred
    rows  = len(indices)
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.2))
    if rows == 1:
        axes = axes[np.newaxis, :]

    fig.patch.set_facecolor('#0f0f0f')
    fig.suptitle('Pedestrian Head Detection — Ground Truth vs Predicted',
                 fontsize=16, fontweight='bold', color='white', y=1.002)

    summary_rows = []

    for row, idx in enumerate(indices):
        img_tensor, target = test_ds[idx]
        img_id = target['image_id']

        # Load original image
        img_path = os.path.join(args.data_root, 'JPEGImages', f'{img_id}.jpg')
        orig     = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H0, W0   = orig.shape[:2]

        # GT boxes → pixel coords
        gt_norm = target['boxes'].numpy()
        gt_px   = gt_norm.copy()
        gt_px[:, [0, 2]] *= W0
        gt_px[:, [1, 3]] *= H0
        n_gt = len(gt_px)

        # Inference
        inp = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(inp)
        preds  = decode_preds(raw, args.img_size, conf_thresh=0.01)
        pb_all = preds[0]['boxes']
        ps_all = preds[0]['scores']
        mask   = ps_all >= args.conf_thresh
        pb_c   = pb_all[mask]
        ps_c   = ps_all[mask]

        # Scale to original
        sx, sy = W0 / args.img_size, H0 / args.img_size
        if len(pb_c):
            pb_c[:, [0, 2]] *= sx
            pb_c[:, [1, 3]] *= sy
        n_pred = len(pb_c)

        #  GT panel 
        ax = axes[row, 0]
        ax.set_facecolor('#0f0f0f')
        ax.imshow(orig)
        for box in gt_px:
            x1, y1, x2, y2 = box
            ax.add_patch(patches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.6, edgecolor='#00e676',
                facecolor='#00e67618',
                boxstyle='square,pad=0'))
        ax.set_title(f'✅  GT  |  {n_gt} heads  |  {img_id}',
                     fontsize=9, color='#00e676', pad=4,
                     fontfamily='monospace')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#00e676')

        #  Pred panel 
        ax = axes[row, 1]
        ax.set_facecolor('#0f0f0f')
        ax.imshow(orig)
        for box, sc in zip(pb_c, ps_c):
            x1, y1, x2, y2 = box
            ax.add_patch(patches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.6, edgecolor='#ff5252',
                facecolor='#ff525218',
                boxstyle='square,pad=0'))
            ax.text(x1 + 2, y1 - 4, f'{sc:.2f}',
                    fontsize=6.5, color='#ffeb3b',
                    bbox=dict(facecolor='#00000080', pad=1, linewidth=0, boxstyle='round'))
        diff  = n_pred - n_gt
        sign  = '+' if diff >= 0 else ''
        color = '#ff5252' if abs(diff) > 2 else '#ffeb3b' if diff != 0 else '#00e676'
        ax.set_title(
            f'🔍  PRED  |  {n_pred} heads  '
            f'(conf≥{args.conf_thresh})  |  diff={sign}{diff}',
            fontsize=9, color=color, pad=4, fontfamily='monospace')
        ax.axis('off')

        summary_rows.append((img_id, n_gt, n_pred, diff))

    plt.tight_layout(h_pad=0.6)
    out = os.path.join(args.save_dir, '01_bbox_comparison.png')
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f' Saved → {out}')
    return summary_rows



# 2. DATASET OVERVIEW


def plot_dataset_overview(args):
    import xml.etree.ElementTree as ET

    split_counts = {}
    all_box_counts = []
    all_areas      = []

    for split in ('train', 'val', 'test'):
        sf = os.path.join(args.data_root, 'ImageSets', 'Main', f'{split}.txt')
        if not os.path.exists(sf):
            continue
        with open(sf) as f:
            ids = [l.strip() for l in f if l.strip()]
        split_counts[split] = len(ids)
        for img_id in ids:
            ap = os.path.join(args.data_root, 'Annotations', f'{img_id}.xml')
            if not os.path.exists(ap):
                all_box_counts.append(0); continue
            root  = ET.parse(ap).getroot()
            boxes = root.findall('object')
            all_box_counts.append(len(boxes))
            sz = root.find('size')
            if sz is not None:
                W = float(sz.find('width').text)
                H = float(sz.find('height').text)
            else:
                W = H = 1.0
            for obj in boxes:
                b  = obj.find('bndbox')
                x1 = float(b.find('xmin').text) / W
                y1 = float(b.find('ymin').text) / H
                x2 = float(b.find('xmax').text) / W
                y2 = float(b.find('ymax').text) / H
                all_areas.append((x2 - x1) * (y2 - y1))

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#111827')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    title_kw   = dict(fontsize=11, color='white', fontweight='bold', pad=8)
    label_kw   = dict(color='#9ca3af', fontsize=9)
    ACCENT     = ['#6366f1', '#22d3ee', '#f59e0b']
    GRID_COLOR = '#1f2937'

    # (a) Split distribution — bar
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(GRID_COLOR)
    splits = list(split_counts.keys())
    counts = list(split_counts.values())
    bars   = ax.bar(splits, counts, color=ACCENT[:len(splits)], width=0.5, zorder=3)
    ax.set_title('Dataset Split Distribution', **title_kw)
    ax.set_ylabel('Image Count', **label_kw)
    ax.tick_params(colors='#9ca3af')
    ax.spines[:].set_color('#374151')
    ax.yaxis.label.set_color('#9ca3af')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(c), ha='center', color='white', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis='y', color='#374151', alpha=0.5, zorder=0)

    total_img = sum(counts)
    total_box = sum(all_box_counts)
    avg_box   = np.mean(all_box_counts) if all_box_counts else 0

    # (b) Pie chart
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(GRID_COLOR)
    wedges, texts, autotexts = ax.pie(
        counts, labels=splits, colors=ACCENT[:len(splits)],
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 9},
        wedgeprops={'linewidth': 1.5, 'edgecolor': '#111827'})
    for at in autotexts:
        at.set_color('white'); at.set_fontsize(9)
    ax.set_title('Split Proportions', **title_kw)

    # (c) Boxes per image histogram
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(GRID_COLOR)
    ax.hist(all_box_counts, bins=30, color='#6366f1', edgecolor='#111827',
            linewidth=0.5, zorder=3)
    ax.axvline(avg_box, color='#f59e0b', lw=1.8, ls='--',
               label=f'mean={avg_box:.1f}')
    ax.set_title('Heads per Image Distribution', **title_kw)
    ax.set_xlabel('Number of heads', **label_kw)
    ax.set_ylabel('Frequency', **label_kw)
    ax.legend(fontsize=8, labelcolor='white', facecolor='#1f2937', edgecolor='none')
    ax.tick_params(colors='#9ca3af')
    ax.spines[:].set_color('#374151')
    ax.grid(color='#374151', alpha=0.4, zorder=0)

    # (d) Box area distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(GRID_COLOR)
    areas_pct = np.array(all_areas) * 100
    ax.hist(areas_pct, bins=40, color='#22d3ee', edgecolor='#111827',
            linewidth=0.5, zorder=3)
    ax.set_title('Bounding Box Area (% of image)', **title_kw)
    ax.set_xlabel('Area %', **label_kw)
    ax.set_ylabel('Frequency', **label_kw)
    ax.tick_params(colors='#9ca3af')
    ax.spines[:].set_color('#374151')
    ax.grid(color='#374151', alpha=0.4, zorder=0)

    # (e) Summary stats table
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(GRID_COLOR)
    ax.axis('off')
    stats = [
        ['Total Images',    f'{total_img:,}'],
        ['Total GT Boxes',  f'{total_box:,}'],
        ['Avg Heads/Image', f'{avg_box:.2f}'],
        ['Max Heads/Image', f'{max(all_box_counts) if all_box_counts else 0}'],
        ['Min Heads/Image', f'{min(all_box_counts) if all_box_counts else 0}'],
        ['Avg Box Area',    f'{np.mean(all_areas)*100:.2f}%' if all_areas else 'N/A'],
        ['Train',           f'{split_counts.get("train",0):,}'],
        ['Val',             f'{split_counts.get("val",0):,}'],
        ['Test',            f'{split_counts.get("test",0):,}'],
    ]
    tbl = ax.table(cellText=stats, colLabels=['Metric', 'Value'],
                   cellLoc='center', loc='center',
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1f2937' if r % 2 == 0 else '#111827')
        cell.set_text_props(color='white' if r > 0 else '#6366f1',
                            fontweight='bold' if r == 0 else 'normal')
        cell.set_edgecolor('#374151')
    ax.set_title('Dataset Statistics', **title_kw)

    # (f) Sample images (3 random)
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(GRID_COLOR)
    ax.axis('off')
    ax.set_title('Sample Images', **title_kw)

    # show 3 tiny samples inside this cell
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 2], wspace=0.05)
    ds_temp = PHDDataset(args.data_root, 'train', 128)
    sample_ids = random.sample(range(len(ds_temp)), min(3, len(ds_temp)))
    for k, sid in enumerate(sample_ids):
        img_t, tgt = ds_temp[sid]
        sax = fig.add_subplot(inner[k])
        sax.imshow(img_t.permute(1,2,0).numpy())
        sax.axis('off')

    fig.suptitle('PHD Dataset Overview', fontsize=16, color='white',
                 fontweight='bold', y=1.01)
    out = os.path.join(args.save_dir, '02_dataset_overview.png')
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {out}')


# 3. RUN FULL INFERENCE

def run_inference(model, args, device):
    test_loader = get_loader(args.data_root, 'test',
                             args.batch_size, args.img_size, args.workers)
    all_preds, all_gts = [], []
    all_scores_flat, all_labels_flat = [], []  # for ROC

    for imgs, targets in tqdm(test_loader, desc='  Running inference', ncols=70):
        imgs        = imgs.to(device)
        raw         = model(imgs)
        batch_preds = decode_preds(raw, args.img_size, conf_thresh=0.01)
        all_preds.extend(batch_preds)
        for tgt in targets:
            gt_boxes = tgt['boxes'].numpy() * args.img_size
            all_gts.append({'boxes': gt_boxes})

    # Build flat score/label for ROC (anchor-level, IoU≥0.5 → positive)
    for preds, gts in zip(all_preds, all_gts):
        pb, ps = preds['boxes'], preds['scores']
        gb     = gts['boxes']
        for box, score in zip(pb, ps):
            all_scores_flat.append(float(score))
            if len(gb):
                ious = box_iou_np(box[None], gb).flatten()
                all_labels_flat.append(1 if ious.max() >= 0.5 else 0)
            else:
                all_labels_flat.append(0)

    return all_preds, all_gts, np.array(all_scores_flat), np.array(all_labels_flat)



# 4. METRICS + CONFUSION MATRIX


def compute_metrics(all_preds, all_gts, conf_thresh=0.5, iou_thresh=0.5):
    tp = fp = fn = 0
    per_img_tp, per_img_fp, per_img_fn = [], [], []

    for preds, gts in zip(all_preds, all_gts):
        pb, ps = preds['boxes'], preds['scores']
        gb     = gts['boxes']
        pb_c   = pb[ps >= conf_thresh]
        matched = set()
        img_tp = img_fp = 0

        for box in pb_c:
            if len(gb) == 0:
                img_fp += 1; continue
            ious = box_iou_np(box[None], gb).flatten()
            best = int(ious.argmax())
            if ious[best] >= iou_thresh and best not in matched:
                img_tp += 1; matched.add(best)
            else:
                img_fp += 1

        img_fn = len(gb) - len(matched)
        tp += img_tp; fp += img_fp; fn += img_fn
        per_img_tp.append(img_tp); per_img_fp.append(img_fp); per_img_fn.append(img_fn)

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)
    tn        = 0   # background is not tracked explicitly
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1,
                per_img_tp=per_img_tp, per_img_fp=per_img_fp, per_img_fn=per_img_fn)


def plot_confusion_matrix(metrics, save_dir):
    tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
    # 2×2: rows = actual, cols = predicted
    cm = np.array([[tp, fn],
                   [fp, 0]])   # TN unknown → 0 shown

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    cmap = LinearSegmentedColormap.from_list('cm', ['#111827', '#6366f1'])
    im   = ax.imshow(cm, cmap=cmap)

    labels = [['TP', 'FN'], ['FP', 'TN*']]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f'{labels[i][j]}\n{val:,}',
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if val > cm.max() * 0.4 else '#9ca3af')

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'],
                       color='white', fontsize=10)
    ax.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'],
                       color='white', fontsize=10)
    ax.set_title('Confusion Matrix  (conf≥0.5, IoU≥0.5)\n'
                 '* TN = background anchors not tracked',
                 color='white', fontsize=11, pad=12)
    plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color='white')
    plt.tight_layout()

    out = os.path.join(save_dir, '04_confusion_matrix.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {out}')



# 5. ROC + AUC


def plot_roc_auc(scores, labels, save_dir):
    from sklearn.metrics import roc_curve, auc
    if len(np.unique(labels)) < 2:
        print('  ROC skipped: only one class in labels.')
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    ax.plot(fpr, tpr, color='#6366f1', lw=2.5,
            label=f'ROC curve  (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#6366f1')
    ax.plot([0, 1], [0, 1], 'w--', lw=1.2, alpha=0.4, label='Random (AUC=0.50)')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate', color='#9ca3af', fontsize=11)
    ax.set_ylabel('True Positive Rate', color='#9ca3af', fontsize=11)
    ax.set_title('ROC Curve — Objectness Score', color='white', fontsize=13,
                 fontweight='bold', pad=12)
    ax.tick_params(colors='#9ca3af')
    ax.spines[:].set_color('#374151')
    ax.grid(color='#374151', alpha=0.35)
    ax.legend(facecolor='#1f2937', edgecolor='none',
              labelcolor='white', fontsize=10)

    # Mark optimal threshold (closest to top-left)
    dist   = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    opt_i  = int(np.argmin(dist))
    ax.scatter(fpr[opt_i], tpr[opt_i], s=100, color='#f59e0b', zorder=5,
               label=f'Optimal  ({fpr[opt_i]:.3f}, {tpr[opt_i]:.3f})')
    ax.legend(facecolor='#1f2937', edgecolor='none',
              labelcolor='white', fontsize=10)

    plt.tight_layout()
    out = os.path.join(save_dir, '05_roc_auc.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {out}')
    return roc_auc


# 6. PR / mAP PLOTS


def plot_metrics_dashboard(all_preds, all_gts, metrics, save_dir):
    iou_thresholds = np.arange(0.5, 1.0, 0.05).round(2).tolist()
    ap_list  = []
    pr_50    = pr_75 = None

    for iou_t in iou_thresholds:
        ap, rec, pre = compute_ap_at_iou(all_preds, all_gts, iou_t)
        ap_list.append(ap)
        if abs(iou_t - 0.50) < 1e-4: pr_50 = (rec, pre, ap)
        if abs(iou_t - 0.75) < 1e-4: pr_75 = (rec, pre, ap)

    map50    = ap_list[0]
    map50_95 = float(np.mean(ap_list))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#111827')
    for ax in axes:
        ax.set_facecolor('#1f2937')
        ax.tick_params(colors='#9ca3af')
        ax.spines[:].set_color('#374151')
        ax.grid(color='#374151', alpha=0.3)

    title_kw = dict(color='white', fontsize=12, fontweight='bold', pad=10)
    label_kw = dict(color='#9ca3af', fontsize=10)

    # (a) PR Curves
    ax = axes[0]
    if pr_50:
        ax.plot(*pr_50[:2], color='#6366f1', lw=2,
                label=f'IoU=0.50  AP={pr_50[2]:.3f}')
        ax.fill_between(pr_50[0], pr_50[1], alpha=0.1, color='#6366f1')
    if pr_75:
        ax.plot(*pr_75[:2], color='#22d3ee', lw=2,
                label=f'IoU=0.75  AP={pr_75[2]:.3f}')
    ax.set_xlabel('Recall', **label_kw); ax.set_ylabel('Precision', **label_kw)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title('Precision-Recall Curve', **title_kw)
    ax.legend(facecolor='#111827', edgecolor='none', labelcolor='white', fontsize=9)

    # (b) AP vs IoU
    ax = axes[1]
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(iou_thresholds)))
    for x, y, c in zip(iou_thresholds, ap_list, colors):
        ax.bar(x, y, width=0.04, color=c, alpha=0.85)
    ax.axhline(map50_95, color='#f59e0b', lw=1.8, ls='--',
               label=f'mAP@[.5:.95]={map50_95:.3f}')
    ax.set_xlabel('IoU Threshold', **label_kw); ax.set_ylabel('AP', **label_kw)
    ax.set_ylim(0, 1); ax.set_title('AP vs IoU Threshold', **title_kw)
    ax.legend(facecolor='#111827', edgecolor='none', labelcolor='white', fontsize=9)

    # (c) Per-image TP/FP/FN bar
    ax  = axes[2]
    ptp = metrics['per_img_tp']; pfp = metrics['per_img_fp']; pfn = metrics['per_img_fn']
    means = [np.mean(ptp), np.mean(pfp), np.mean(pfn)]
    stds  = [np.std(ptp),  np.std(pfp),  np.std(pfn)]
    clrs  = ['#22d3ee', '#f87171', '#f59e0b']
    bars  = ax.bar([0, 1, 2], means, yerr=stds, capsize=6,
                   color=clrs, alpha=0.85, width=0.5, error_kw={'ecolor': 'white'})
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['TP', 'FP', 'FN'], color='white', fontsize=11)
    ax.set_ylabel('Count per image (mean ± std)', **label_kw)
    ax.set_title('Per-Image Detection Stats', **title_kw)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{m:.1f}', ha='center', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(save_dir, '03_metrics_dashboard.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f' Saved → {out}')
    return map50, map50_95

# 7. METRICS TABLE (pretty)


def plot_metrics_table(metrics, map50, map50_95, roc_auc, save_dir):
    rows = [
        ['Precision',           f'{metrics["precision"]:.4f}'],
        ['Recall',              f'{metrics["recall"]:.4f}'],
        ['F1-Score',            f'{metrics["f1"]:.4f}'],
        ['mAP@0.5',             f'{map50:.4f}'],
        ['mAP@[0.5:0.95]',      f'{map50_95:.4f}'],
        ['ROC-AUC',             f'{roc_auc:.4f}' if roc_auc else 'N/A'],
        ['True Positives',      f'{metrics["tp"]:,}'],
        ['False Positives',     f'{metrics["fp"]:,}'],
        ['False Negatives',     f'{metrics["fn"]:,}'],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#111827')
    ax.axis('off')
    ax.set_facecolor('#111827')

    tbl = ax.table(cellText=rows, colLabels=['Metric', 'Value'],
                   cellLoc='center', loc='center',
                   bbox=[0.05, 0.02, 0.9, 0.9])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#6366f1')
            cell.set_text_props(color='white', fontweight='bold', fontsize=12)
        else:
            cell.set_facecolor('#1f2937' if r % 2 else '#111827')
            cell.set_text_props(color='white' if c == 0 else '#22d3ee', fontsize=12)
        cell.set_edgecolor('#374151')

    ax.set_title('Performance Metrics — YoloX+CFE on PHD Test Set',
                 color='white', fontsize=13, fontweight='bold', pad=18)
    plt.tight_layout()
    out = os.path.join(save_dir, '06_metrics_table.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved → {out}')


# MAIN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   type=str, required=True)
    p.add_argument('--checkpoint',  type=str, required=True)
    p.add_argument('--save_dir',    type=str, default='./results')
    p.add_argument('--model_size',  type=str, default='s', choices=['s', 'm', 'l'])
    p.add_argument('--img_size',    type=int, default=416)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--workers',     type=int, default=2)
    p.add_argument('--n_images',    type=int, default=10,
                   help='Number of bbox visualisation images (min 10)')
    p.add_argument('--conf_thresh', type=float, default=0.5)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    print(f'\n YoloX+CFE — Full Results Visualiser')
    print(f'   device={device}  save_dir={args.save_dir}\n')

    # Load model
    model = YoloXCFE(args.model_size, nc=1).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint epoch {ckpt["epoch"]+1}  '
          f'(val_loss={ckpt["best_loss"]:.4f})\n')

    test_ds = PHDDataset(args.data_root, 'test', args.img_size)

    print('[1/6] Generating bounding-box comparison grid …')
    summary = plot_bbox_grid(test_ds, model, device, args)
    print(f'       → {len(summary)} images, '
          f'diffs: {[r[3] for r in summary]}\n')

    print('[2/6] Generating dataset overview …')
    plot_dataset_overview(args)
    print()

    print('[3/6] Running full test-set inference …')
    all_preds, all_gts, scores_flat, labels_flat = run_inference(model, args, device)
    print()

    print('[4/6] Computing metrics & plotting dashboard …')
    metrics = compute_metrics(all_preds, all_gts,
                              conf_thresh=args.conf_thresh, iou_thresh=0.5)
    map50, map50_95 = plot_metrics_dashboard(all_preds, all_gts, metrics, args.save_dir)
    print()

    print('[5/6] Plotting confusion matrix …')
    plot_confusion_matrix(metrics, args.save_dir)
    print()

    print('[6/6] Plotting ROC / AUC …')
    roc_auc = None
    try:
        import sklearn  # noqa
        roc_auc = plot_roc_auc(scores_flat, labels_flat, args.save_dir)
    except ImportError:
        print(' scikit-learn not found. Install with: pip install scikit-learn')
    print()

    print('Saving metrics table …')
    plot_metrics_table(metrics, map50, map50_95, roc_auc, args.save_dir)

    # ── Console summary ──────────────────────────────────────────────────
    print()
    print('=' * 58)
    print('  Final Results  — YoloX+CFE on PHD Test Set')
    print('=' * 58)
    print(f'  Precision       : {metrics["precision"]:.4f}')
    print(f'  Recall          : {metrics["recall"]:.4f}')
    print(f'  F1-Score        : {metrics["f1"]:.4f}')
    print(f'  mAP@0.5         : {map50:.4f}')
    print(f'  mAP@[0.5:0.95]  : {map50_95:.4f}')
    if roc_auc:
        print(f'  ROC-AUC         : {roc_auc:.4f}')
    print(f'  TP / FP / FN    : {metrics["tp"]} / {metrics["fp"]} / {metrics["fn"]}')
    print('=' * 58)
    print(f'\nAll plots saved in → {args.save_dir}/')
    print('   01_bbox_comparison.png')
    print('   02_confusion_matrix.png')
    print('   03_roc_auc.png')


if __name__ == '__main__':
    main()
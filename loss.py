import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou(b1, b2):
    ix1 = torch.max(b1[:, 0], b2[:, 0])
    iy1 = torch.max(b1[:, 1], b2[:, 1])
    ix2 = torch.min(b1[:, 2], b2[:, 2])
    iy2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    return inter / (a1 + a2 - inter + 1e-7)


def ciou_loss(pred, gt):
    iou  = box_iou(pred, gt)
    cx_p = (pred[:, 0] + pred[:, 2]) / 2
    cy_p = (pred[:, 1] + pred[:, 3]) / 2
    cx_g = (gt[:, 0]   + gt[:, 2])   / 2
    cy_g = (gt[:, 1]   + gt[:, 3])   / 2
    d2   = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2
    ex1  = torch.min(pred[:, 0], gt[:, 0])
    ey1  = torch.min(pred[:, 1], gt[:, 1])
    ex2  = torch.max(pred[:, 2], gt[:, 2])
    ey2  = torch.max(pred[:, 3], gt[:, 3])
    c2   = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7
    wp   = (pred[:, 2] - pred[:, 0]).clamp(1e-3)
    hp   = (pred[:, 3] - pred[:, 1]).clamp(1e-3)
    wg   = (gt[:, 2]   - gt[:, 0]).clamp(1e-3)
    hg   = (gt[:, 3]   - gt[:, 1]).clamp(1e-3)
    v    = (4 / torch.pi ** 2) * (torch.atan(wg / hg) - torch.atan(wp / hp)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    return 1 - iou + d2 / c2 + alpha * v


def decode(raw_outputs, img_size=640):
    strides = [img_size // o.shape[2] for o in raw_outputs]
    all_p   = []
    for out, stride in zip(raw_outputs, strides):
        B, C, H, W = out.shape
        gy, gx = torch.meshgrid(
            torch.arange(H, device=out.device),
            torch.arange(W, device=out.device),
            indexing='ij',
        )
        grid  = torch.stack([gx, gy], -1).float().reshape(1, H * W, 2)
        out   = out.permute(0, 2, 3, 1).reshape(B, H * W, C)
        xy    = (out[..., :2].sigmoid() + grid) * stride
        wh    = out[..., 2:4].exp() * stride
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], -1)
        obj   = out[..., 4:5]
        cls   = out[..., 5:]
        all_p.append(torch.cat([boxes, obj, cls], -1))
    return torch.cat(all_p, 1)


class YoloXLoss(nn.Module):
    def __init__(self, img_size=640, nc=1):
        super().__init__()
        self.img_size = img_size
        self.nc       = nc

    def forward(self, raw_outputs, targets):
        B   = raw_outputs[0].shape[0]
        dev = raw_outputs[0].device
        preds = decode(raw_outputs, self.img_size)
        A     = preds.shape[1]

        loss_iou = torch.tensor(0., device=dev)
        loss_obj = torch.tensor(0., device=dev)
        loss_cls = torch.tensor(0., device=dev)
        obj_tgt  = torch.zeros(B, A, device=dev)

        for i in range(B):
            gt_b = targets[i]['boxes'].to(dev) * self.img_size
            N    = len(gt_b)
            if N == 0:
                continue
            pb = preds[i, :, :4]
            k  = min(10, A)

            for j in range(N):
                iou_row = box_iou(pb, gt_b[j:j + 1].expand_as(pb))
                idx     = iou_row.topk(k).indices
                obj_tgt[i, idx] = 1.

                loss_iou = loss_iou + ciou_loss(
                    pb[idx], gt_b[j:j + 1].expand(k, 4)
                ).mean()

                cls_pred = preds[i, idx, 5:]
                cls_tgt  = torch.zeros_like(cls_pred)
                cls_tgt[:, 0] = 1.
                loss_cls = loss_cls + F.binary_cross_entropy_with_logits(cls_pred, cls_tgt)

        raw_obj = torch.cat(
            [o.permute(0, 2, 3, 1).reshape(B, -1, o.shape[1])[:, :, 4:5]
             for o in raw_outputs], 1
        ).squeeze(-1)
        loss_obj = F.binary_cross_entropy_with_logits(raw_obj, obj_tgt)

        total = 5 * loss_iou + loss_obj + loss_cls
        return total, {
            'iou': loss_iou.item(),
            'obj': loss_obj.item(),
            'cls': loss_cls.item(),
        }
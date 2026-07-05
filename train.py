"""
train.py — Train YoloX+CFE on the PHD (Pedestrian Head Detection) dataset.

Usage:
    python train.py --data_root /path/to/VOC2007 --save_dir ./runs
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_loader
from model   import YoloXCFE
from loss    import YoloXLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to VOC2007 dataset root')
    p.add_argument('--save_dir',  type=str, default='./runs',
                   help='Directory to save checkpoints')
    p.add_argument('--model_size', type=str, default='s',
                   choices=['s', 'm', 'l'], help='YoloX backbone size')
    p.add_argument('--epochs',     type=int,   default=5)
    p.add_argument('--batch_size', type=int,   default=4)
    p.add_argument('--img_size',   type=int,   default=416)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--workers',    type=int,   default=2)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    train_loader = get_loader(args.data_root, 'train',
                              args.batch_size, args.img_size, args.workers)
    val_loader   = get_loader(args.data_root, 'val',
                              args.batch_size, args.img_size, args.workers)

    # Model
    model     = YoloXCFE(args.model_size, nc=1).to(device)
    criterion = YoloXLoss(args.img_size, nc=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        #  Train 
        model.train()
        total_loss = 0.
        for step, (imgs, tgts) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outs         = model(imgs)
            loss, parts  = criterion(outs, tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0:
                print(f'  E{epoch+1:3d} [{step:4d}/{len(train_loader)}] '
                      f'loss={loss.item():.3f} '
                      f'iou={parts["iou"]:.3f} '
                      f'obj={parts["obj"]:.3f}')

        scheduler.step()

        #  Validate 
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = imgs.to(device)
                loss, _ = criterion(model(imgs), tgts)
                val_loss += loss.item()

        val_loss   /= len(val_loader)
        train_loss  = total_loss / len(train_loader)
        print(f'[Epoch {epoch+1}/{args.epochs}] '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.6f}')

        #  Checkpoint 
        ckpt = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pth'))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
            print(f'  Best model saved! (val_loss={best_loss:.4f})')

    print(f'\n Training complete! Best val loss: {best_loss}')


if __name__ == '__main__':
    main()
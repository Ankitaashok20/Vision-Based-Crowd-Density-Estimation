import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader


class PHDDataset(Dataset):
    def __init__(self, root, split='train', img_size=640):
        self.root     = root
        self.split    = split
        self.img_size = img_size
        self.img_dir  = os.path.join(root, 'JPEGImages')
        self.ann_dir  = os.path.join(root, 'Annotations')
        split_file    = os.path.join(root, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        print(f'[PHDDataset] split={split}  images={len(self.ids)}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        img      = cv2.imread(img_path)
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0   = img.shape[:2]
        ann_path = os.path.join(self.ann_dir, f'{img_id}.xml')
        boxes    = self._parse_xml(ann_path, w0, h0)
        img      = cv2.resize(img, (self.img_size, self.img_size))
        img      = torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1)
        return img, {
            'boxes':    torch.tensor(boxes, dtype=torch.float32),
            'labels':   torch.zeros(len(boxes), dtype=torch.long),
            'image_id': img_id
        }

    def _parse_xml(self, path, w, h):
        if not os.path.exists(path):
            return []
        root = ET.parse(path).getroot()
        boxes = []
        for obj in root.findall('object'):
            b = obj.find('bndbox')
            boxes.append([
                float(b.find('xmin').text) / w,
                float(b.find('ymin').text) / h,
                float(b.find('xmax').text) / w,
                float(b.find('ymax').text) / h,
            ])
        return boxes


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


def get_loader(data_root, split, batch_size=8, img_size=640, num_workers=2):
    ds = PHDDataset(data_root, split, img_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
"""RT-DETRv2 を Gazebo で生成した俯瞰カメラ画像で学習する（ADR 0008）.

pixi run --frozen train-detector --config configs/detection/train_rtdetr.yaml
出力: <output_dir>（save_pretrained 形式。val loss 最小のもの）と train_info.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from pai_training.detection.coco import flip_target, split_ids, to_target


class CocoDetection(Dataset):
    def __init__(self, items, processor, augment=None):
        self.items = items              # [(image_path, annotations, width, height)]
        self.processor = processor
        self.augment = augment
        if augment:
            self.color = v2.ColorJitter(augment['brightness'], augment['contrast'], augment['saturation'],
                                        augment['hue'])
            self.blur = v2.GaussianBlur(5, sigma=(0.3, 1.5))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, anns, w, h = self.items[i]
        img = Image.open(path).convert('RGB')
        target = to_target(anns, w, h)
        if self.augment:
            a = self.augment
            img = self.color(img)
            if random.random() < a['blur_prob']:
                img = self.blur(img)
            if random.random() < a['flip_prob']:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                target = flip_target(target)
        pixel = self.processor(images=img, return_tensors='pt')['pixel_values'][0]
        if self.augment and self.augment['noise_std'] > 0:
            pixel = (pixel + torch.randn_like(pixel) * self.augment['noise_std']).clamp(0, 1)
        return pixel, target


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def load_items(dirs):
    items = []
    for d in dirs:
        d = Path(d).expanduser()
        coco = json.loads((d / 'annotations.json').read_text())
        anns = {}
        for a in coco['annotations']:
            anns.setdefault(a['image_id'], []).append(a)
        for im in coco['images']:
            items.append((str(d / 'images' / im['file_name']), anns.get(im['id'], []), im['width'], im['height']))
    return items


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for pixel, labels in loader:
            out = model(pixel_values=pixel.to(device), labels=[{k: v.to(device) for k, v in t.items()} for t in labels])
            total += out.loss.item() * len(labels)
            n += len(labels)
    model.train()
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    items = load_items(cfg['dataset_dirs'])
    tr_idx, va_idx = split_ids(range(len(items)), cfg['val_ratio'], cfg['seed'])
    size = cfg['image_size']
    processor = RTDetrImageProcessor.from_pretrained(cfg['model'], size={'height': size, 'width': size})
    train = CocoDetection([items[i] for i in tr_idx], processor, cfg['augment'])
    val = CocoDetection([items[i] for i in va_idx], processor, None)
    tl = DataLoader(train, cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=collate)
    vl = DataLoader(val, cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], collate_fn=collate)
    print(f'train {len(train)} / val {len(val)} images')

    id2label = dict(enumerate(cfg['labels']))
    model = RTDetrV2ForObjectDetection.from_pretrained(
        cfg['model'], id2label=id2label, label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True).to(device)
    backbone = [p for n, p in model.named_parameters() if 'backbone' in n]
    others = [p for n, p in model.named_parameters() if 'backbone' not in n]
    opt = torch.optim.AdamW([{'params': others, 'lr': cfg['lr']}, {'params': backbone, 'lr': cfg['lr_backbone']}],
                            weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'] * len(tl))

    out = Path(cfg['output_dir']).expanduser()
    best, history, t0 = float('inf'), [], time.time()
    model.train()
    for epoch in range(cfg['epochs']):
        running = 0.0
        for pixel, labels in tl:
            outputs = model(pixel_values=pixel.to(device),
                            labels=[{k: v.to(device) for k, v in t.items()} for t in labels])
            opt.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            running += outputs.loss.item()
        val_loss = evaluate(model, vl, device)
        history.append({'epoch': epoch, 'train_loss': running / len(tl), 'val_loss': val_loss})
        print(f'epoch {epoch}: train {running / len(tl):.4f} val {val_loss:.4f}', flush=True)
        if val_loss < best:
            best = val_loss
            model.save_pretrained(out)
            processor.save_pretrained(out)
    info = {'config': args.config, 'best_val_loss': best, 'train_seconds': time.time() - t0,
            'train_images': len(train), 'val_images': len(val), 'val_ids': va_idx, 'history': history}
    (out / 'train_info.json').write_text(json.dumps(info, indent=2))
    print(f'saved {out} (best val loss {best:.4f})')


if __name__ == '__main__':
    main()

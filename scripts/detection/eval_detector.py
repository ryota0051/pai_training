"""物体検出器の位置推定精度を評価し、赤色検出（object_detector_node と同じ処理）と比べる.

pixi run --frozen eval-detector --config configs/detection/eval_rtdetr.yaml [--dataset-dir ...] [--label ...]
出力: <output_dir>/<label>.csv（画像ごとの誤差）と <label>_summary.json
"""
import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection


def summarize(errors_mm, n):
    e = np.asarray([x for x in errors_mm if x is not None])
    out = {'images': n, 'detected': int(len(e)), 'detection_rate': len(e) / n if n else 0.0}
    if len(e):
        out.update({'mean_mm': float(e.mean()), 'median_mm': float(np.median(e)),
                    'p95_mm': float(np.percentile(e, 95)), 'within_5mm': float((e <= 5).mean() * len(e) / n),
                    'within_10mm': float((e <= 10).mean() * len(e) / n)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset-dir', default=None)
    parser.add_argument('--label', default='dr_test')
    parser.add_argument('--examples', type=int, default=0, help='検出結果を描いた画像を先頭から何枚並べて保存するか')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sys.path.insert(0, cfg['pai_task_env_path'])
    from pai_task_env.domain_randomization import bbox_center_to_position
    from pai_task_env.vision import CameraModel, mask_centroid, red_mask

    cam_all = yaml.safe_load(open(cfg['camera_config']))['object_detector_node']['ros__parameters']
    c = cam_all['camera']
    cam = CameraModel.from_hfov(c['width'], c['height'], c['hfov'], c['position'], c['rpy'])
    size = (c['width'], c['height'])
    det = cam_all['detection']
    r, h, zc = cfg['object_size'] / 2, cfg['object_size'], cfg['object_center_z']

    device = torch.device(cfg['device'])
    processor = RTDetrImageProcessor.from_pretrained(cfg['model_dir'])
    model = RTDetrV2ForObjectDetection.from_pretrained(cfg['model_dir']).to(device).eval()

    d = Path(args.dataset_dir or cfg['dataset_dir']).expanduser()
    coco = json.loads((d / 'annotations.json').read_text())
    meta = {m['image_id']: m for m in map(json.loads, open(d / 'meta.jsonl'))}
    rows, examples = [], []
    for im in coco['images']:
        m = meta[im['id']]
        obj = np.asarray(m['object'][:2])
        img = Image.open(d / 'images' / im['file_name']).convert('RGB')
        # 学習した検出器: 最もスコアの高い枠 → 側面の見え方を補正して位置を逆算
        with torch.no_grad():
            inputs = processor(images=img, return_tensors='pt').to(device)
            out = model(**inputs)
        res = processor.post_process_object_detection(out, threshold=cfg['score_threshold'],
                                                      target_sizes=[(img.height, img.width)])[0]
        err_det, score = None, None
        if len(res['scores']):
            k = int(res['scores'].argmax())
            u0, v0, u1, v1 = res['boxes'][k].tolist()
            xy = bbox_center_to_position(cam, ((u0 + u1) / 2, (v0 + v1) / 2), r, h, size, center_z=zc)
            err_det, score = float(np.linalg.norm(np.asarray(xy) - obj) * 1000), float(res['scores'][k])
        # 比較: 赤色検出（object_detector_node と同じ: 赤画素の重心を円柱上面の高さへ投影）
        rgb = np.asarray(img)
        cen = mask_centroid(red_mask(rgb, det['min_red'], det['max_other_ratio']), det['min_pixels'])
        err_red = None
        if cen is not None:
            p = cam.pixel_to_plane(cen[0], cen[1], cam_all['plane_z'])
            err_red = float(np.linalg.norm(np.asarray(p[:2]) - obj) * 1000)
        if len(examples) < args.examples:
            # 黄: 学習した検出器の枠、赤い×: 赤色検出の推定位置（検出できなければ描かない）
            vis = img.copy()
            dr = ImageDraw.Draw(vis)
            if err_det is not None:
                dr.rectangle([u0, v0, u1, v1], outline=(255, 230, 0), width=2)
            if cen is not None:
                cu, cv_ = cen
                dr.line([cu - 6, cv_ - 6, cu + 6, cv_ + 6], fill=(255, 40, 40), width=3)
                dr.line([cu - 6, cv_ + 6, cu + 6, cv_ - 6], fill=(255, 40, 40), width=3)
            examples.append(vis)
        hue = colorsys.rgb_to_hsv(*m['scene']['object_rgb'])[0]
        rows.append({'image_id': im['id'], 'err_detector_mm': err_det, 'score': score, 'err_red_mm': err_red,
                     'object_hue': hue, 'distractors': len(m['scene']['distractors'])})

    out_dir = Path(cfg['output_dir']).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_dir / f'{args.label}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    redish = [r_ for r_ in rows if r_['object_hue'] < 0.05 or r_['object_hue'] > 0.95]
    summary = {
        'label': args.label, 'model_dir': cfg['model_dir'], 'dataset_dir': str(d),
        'detector': summarize([r_['err_detector_mm'] for r_ in rows], len(rows)),
        'red_threshold': summarize([r_['err_red_mm'] for r_ in rows], len(rows)),
        'red_threshold_on_reddish_objects': summarize([r_['err_red_mm'] for r_ in redish], len(redish)),
    }
    (out_dir / f'{args.label}_summary.json').write_text(json.dumps(summary, indent=2))
    if examples:
        cols = 4
        w, h = examples[0].size
        grid = Image.new('RGB', (cols * w, ((len(examples) + cols - 1) // cols) * h), 'white')
        for i, e in enumerate(examples):
            grid.paste(e, ((i % cols) * w, (i // cols) * h))
        grid.save(out_dir / f'{args.label}_examples.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

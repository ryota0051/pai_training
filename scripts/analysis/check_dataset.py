"""変換した LeRobotDataset の全エピソードを数値で検査する（rerun での目視確認の補助）.

pixi run --frozen check-dataset --dataset-root ~/dataset/lerobot_dataset/<name> [--place-xy 0.10 -0.12] [--split 117]
検査項目:
- lag: 指令（action）に関節角（state）が何フレーム遅れて追従するか（誤差が最小になるずらし量）
- still_mm: 把持の指令までに物体が動いた量（物体位置が観測に正しく入っているか）
- tail: 末尾 20 フレームの指令の変化（短いエピソードを最終フレームで延長した部分が一定か）
- gripper_lag: グリッパの閉指令から関節角が閉じ始めるまでのフレーム数
- end_dist_place_mm: 最後の物体位置と置き場所の中心の距離
- 開始姿勢のばらつき（--split で前半・後半を分けて比べる）
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--place-xy', type=float, nargs=2, default=[0.10, -0.12],
                        help='置き場所の中心（pai_task_env/config/task.yaml の place.xy）')
    parser.add_argument('--close-threshold', type=float, default=0.12, help='グリッパ指令が閉とみなす値 [rad]')
    parser.add_argument('--split', type=int, default=None, help='この番号の前後で開始姿勢のばらつきを比べる')
    args = parser.parse_args()
    files = sorted(glob.glob(str(Path(args.dataset_root).expanduser() / 'data' / 'chunk-*' / '*.parquet')))
    rows = []
    for f in files:
        d = pd.read_parquet(f)
        s, a = np.stack(d['observation.state']), np.stack(d['action'])
        e = np.stack(d['observation.environment_state'])
        g = a[:, -1]
        close = int(np.argmax(g > args.close_threshold)) if (g > args.close_threshold).any() else -1
        lag = min(range(15), key=lambda k: np.abs(s[k:, :-1] - a[:len(a) - k, :-1]).mean())
        gs = s[:, -1]
        opened = int(np.argmax(gs > gs[0] + 0.05)) if (gs > gs[0] + 0.05).any() else -1
        rows.append({
            'episode': int(d['episode_index'].iloc[0]), 'frames': len(d), 'lag': lag,
            'still_mm': float(np.abs(e[:max(close, 1)] - e[0]).max() * 1000),
            'tail': float(np.abs(np.diff(a[-20:], axis=0)).max()),
            'gripper_lag': opened - close,
            'end_dist_place_mm': float(np.linalg.norm(e[-1, :2] - np.asarray(args.place_xy)) * 1000),
            'start': s[0, :-1],
        })
    r = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(f'{len(r)} episodes')
    print(r[['lag', 'still_mm', 'tail', 'gripper_lag', 'end_dist_place_mm']].describe().round(3).to_string())
    starts = np.stack(r['start'])
    if args.split:
        print('開始姿勢の標準偏差 [rad] 前半:', starts[:args.split].std(axis=0).round(4),
              ' 後半:', starts[args.split:].std(axis=0).round(4))
    print('物体が最も動いたエピソード:\n', r.nlargest(3, 'still_mm')[['episode', 'still_mm']].to_string(index=False))
    print('置き場所から最も遠いエピソード:\n', r.nlargest(3, 'end_dist_place_mm')[['episode', 'end_dist_place_mm']].to_string(index=False))


if __name__ == '__main__':
    main()

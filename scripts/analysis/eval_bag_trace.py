"""評価中の rosbag（scripts/sim/eval_positions.sh）から試行ごとの手先軌跡と把持時のずれを取り出す.

pixi run --frozen eval-trace --config configs/analysis/eval_trace.yaml --bag ~/dataset/eval_bags/<label> \
    --eval-csv ~/dataset/eval/<label>_<日時>.csv
出力: <output_dir>/<label>/trace.csv（試行ごとの集計）, traces.npz, trace.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

from pai_training.analysis.trials import enabled_intervals, slice_by_time


def read_bag(path, joint_order):
    js_t, js, obj_t, obj, st_t, st = [], [], [], [], [], []
    cmd_t, cmd = [], []
    with Reader(path) as reader:
        for conn, ts, raw in reader.messages():
            msg = deserialize_cdr(raw, conn.msgtype)
            t = ts * 1e-9
            if conn.topic == '/joint_states':
                pos = dict(zip(msg.name, msg.position))
                if all(n in pos for n in joint_order):
                    js_t.append(t)
                    js.append([pos[n] for n in joint_order])
            elif conn.topic == '/task/object_pose':
                p = msg.pose.position
                obj_t.append(t)
                obj.append([p.x, p.y, p.z])
            elif conn.topic == '/policy/status':
                st_t.append(t)
                st.append(bool(json.loads(msg.data).get('enabled')))
            elif conn.topic == '/crane_plus_arm_controller/joint_trajectory' and msg.points:
                cmd_t.append(t)
                cmd.append(list(msg.points[0].positions))
    a = lambda x: np.asarray(x, dtype=float)
    return (a(js_t), a(js), a(obj_t), a(obj), a(st_t), np.asarray(st), a(cmd_t), a(cmd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--eval-csv', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sys.path.insert(0, cfg['kinematics_package'])
    from crane_plus_kinematics.kinematics import load_kinematics
    kin = load_kinematics(cfg['kinematics_config'])

    js_t, js, obj_t, obj, st_t, st, cmd_t, cmd = read_bag(args.bag, cfg['joint_order'])
    intervals = enabled_intervals(st_t, st)
    ev = pd.read_csv(args.eval_csv)
    if len(intervals) != len(ev):
        print(f'warning: {len(intervals)} enabled intervals vs {len(ev)} trials')

    rows, traces = [], []
    for i, (t0, t1) in enumerate(intervals[:len(ev)]):
        tt, q = slice_by_time(js_t, js, t0, t1)
        _, o = slice_by_time(obj_t, obj, t0, t1)
        _, c = slice_by_time(cmd_t, cmd, t0, t1)
        o0 = o[0]
        ee = np.array([[p.x, p.y, p.z] for p in map(kin.forward, q[:, :4])])
        g = q[:, cfg['gripper_index']]
        idx = np.flatnonzero(g > cfg['close_threshold'])
        k = int(idx[0]) if len(idx) else int(np.argmin(np.linalg.norm(ee - o0, axis=1)))
        # 下降開始（手先が接近高さを下回った瞬間）の手先 y
        low = np.flatnonzero(ee[:, 2] < cfg['approach_z'])
        kd = int(low[0]) if len(low) else k
        d = ee[k] - o0
        rows.append(dict(trial=i, object_x=o0[0], object_y=o0[1], success=bool(ev.success[i]),
                         closed=bool(len(idx)), t_close=tt[k] - t0, ee_x=ee[k, 0], ee_y=ee[k, 1], ee_z=ee[k, 2],
                         dx=d[0], dy=d[1], dz=d[2], dist=float(np.linalg.norm(d)),
                         j1_close=q[k, 0], dy_descent_start=ee[kd, 1] - o0[1],
                         object_moved=float(np.linalg.norm(o[k, :2] - o0[:2])) if len(o) > k else np.nan))
        traces.append(dict(t=tt - t0, q=q, ee=ee, obj=o0, cmd=c))

    df = pd.DataFrame(rows)
    out = Path(cfg['output_dir']) / Path(args.bag).name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'trace.csv', index=False)
    np.savez_compressed(out / 'traces.npz', **{f'{k}_{i}': v for i, tr in enumerate(traces)
                                               for k, v in tr.items()})
    pd.set_option('display.width', 220)
    print(df.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r, tr in zip(rows, traces):
        color = 'tab:blue' if r['success'] else 'tab:red'
        n = int(np.searchsorted(tr['t'], cfg['plot_seconds']))
        axes[0].plot(tr['ee'][:n, 1], tr['ee'][:n, 0], color=color, lw=0.8)
        axes[0].plot(tr['obj'][1], tr['obj'][0], 'o', color=color, ms=4)
        axes[1].plot(tr['t'][:n], tr['ee'][:n, 1] - tr['obj'][1], color=color, lw=0.8)
    axes[0].set_xlabel('y [m]'); axes[0].set_ylabel('x [m]'); axes[0].invert_xaxis()
    axes[0].set_title('EE path (top view, blue=success, red=fail)')
    axes[1].set_xlabel('t [s]'); axes[1].set_ylabel('ee_y - object_y [m]')
    axes[1].set_title('Lateral error over time')
    fig.tight_layout()
    fig.savefig(out / 'trace.png', dpi=120)
    print(f'saved to {out}')


if __name__ == '__main__':
    main()

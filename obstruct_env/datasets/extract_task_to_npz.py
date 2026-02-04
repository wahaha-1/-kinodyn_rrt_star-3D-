#!/usr/bin/env python3
"""
从单图多任务数据包（map_XXXXX.npz）抽取一个任务，转换为验证脚本兼容的 NPZ（与 plan_and_export 输出一致）。

用法：
  python3 obstruct_env/datasets/extract_task_to_npz.py --dataset obstruct_env/datasets/out/dataset/map_00001.npz --task 0 --out obstruct_env/examples/scenes/generated_traj.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description='从数据包抽取单个任务为验证脚本兼容格式')
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--out', type=str, default='obstruct_env/examples/scenes/generated_traj.npz')
    args = ap.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    tasks = list(data['tasks'])
    t = tasks[args.task]

    # numpy 存的对象数组，需 .item() 取出 dict
    if isinstance(t, np.ndarray) and t.shape == ():
        t = t.item()

    traj = t['result']['trajectory']
    # 兼容 np.savez 中 dict 变 0-d object 的情况
    if isinstance(traj, np.ndarray) and traj.shape == ():
        traj = traj.item()

    out = {
        'trajectory': traj,
        'global_waypoints': t['result']['global_waypoints'],
        'time_steps': t['result']['time_steps'],
        'local_segments': t['result']['local_segments'],
        'raw': {
            'coefficients': t['result']['coefficients'],
            'intervals': t['result']['intervals'],
            'success': t['result']['success'],
            'cost': t['result']['cost'],
        },
        'obstacle_map': data['obstacle_map'],
        'map_info': {
            'grid_res': float(data['resolution']),
            'grid_size': np.asarray(data['obstacle_map']).shape,
            'map_size': data['map_size'],
            'base': str(Path(args.dataset).with_suffix('').name),
        },
        'start_state': t['start_state'],
        'goal_state': t['goal_state'],
        'terminal_attitude': t['result'].get('terminal_attitude', None),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f'已导出到: {out_path}')


if __name__ == '__main__':
    main()

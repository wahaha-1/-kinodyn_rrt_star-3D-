#!/usr/bin/env python3
"""
批量数据集生成：地图 → ESDF → 多任务规划 → 验证与打包（单图多任务）

- 依赖：已有的 C++ 规划器二进制、cpp_bridge、ObstacleMapGenerator
- 输出：每张地图一个 .npz，内含地图与多个任务；同时导出 maps/<base>_occ.npy 与 <base>_esdf.npy

示例：
  python3 obstruct_env/datasets/generate_dataset.py \
      --config obstruct_env/datasets/dataset_config.yaml \
      --out-dir obstruct_env/datasets/out \
      --num-maps 1 --tasks-per-map 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 允许脚本直接运行
import sys
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from cpp_bridge import CppKinoRRTStarBridge
from obstruct_env.core.obstacle_map import ObstacleMapGenerator

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        cfg_path = _repo_root / 'obstruct_env' / 'datasets' / 'dataset_config.yaml'
    else:
        cfg_path = Path(path)
    if cfg_path.suffix.lower() in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError('需要 PyYAML: pip install pyyaml')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif cfg_path.suffix.lower() == '.json':
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError('仅支持 YAML/JSON 配置文件')


def _is_range(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)


def sample_value(spec, *, as_int: bool = False, default=None):
    """通用采样：
    - 标量：返回原值
    - [min,max]：在闭区间均匀采样
    - [a,b,c...]：从列表均匀抽取
    - {dist:'uniform',min:...,max:...}：预留扩展
    """
    if spec is None:
        return default
    if isinstance(spec, dict):
        d = (spec.get('dist') or 'uniform').lower()
        vmin = spec.get('min', None); vmax = spec.get('max', None)
        if d == 'uniform' and vmin is not None and vmax is not None:
            r = np.random.uniform(float(vmin), float(vmax))
            return int(round(r)) if as_int else float(r)
        return spec
    if isinstance(spec, (list, tuple)):
        if _is_range(spec):
            r = np.random.uniform(float(spec[0]), float(spec[1]))
            return int(round(r)) if as_int else float(r)
        choice = random.choice(list(spec))
        return int(choice) if as_int and isinstance(choice, (int, float)) else choice
    if as_int and isinstance(spec, (float, int)):
        return int(spec)
    return spec


def _ensure_dirs(out_dir: Path) -> Tuple[Path, Path, Path]:
    maps_dir = out_dir / 'maps'
    ds_dir = out_dir / 'dataset'
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, maps_dir, ds_dir


def _gen_one_map(gen: ObstacleMapGenerator, env_cfg: Dict[str, Any]):
    env = (env_cfg.get('environment') or 'city').lower()
    if env == 'city':
        bsec = env_cfg.get('boxes', {})
        cnt = sample_value(bsec.get('count', 15), as_int=True)
        size_rng = bsec.get('size_range', [2.0, 5.0])
        gen.generate_random_boxes(num_obstacles=cnt, size_range=tuple(size_rng), safe_margin=2.0)
    elif env == 'buildings':
        bsec = env_cfg.get('buildings', {})
        cnt = sample_value(bsec.get('count', 20), as_int=True)
        footprint_rng = bsec.get('footprint_range', [2.0, 6.0])
        height_rng = bsec.get('height_range', [3.0, 10.0])
        min_gap = float(bsec.get('min_gap', 1.0))
        boundary_margin = float(bsec.get('boundary_margin', 1.0))
        max_tries = int(bsec.get('max_tries_per_building', 200))
        gen.generate_building_city_environment(
            num_buildings=int(cnt),
            footprint_range=tuple(footprint_rng),
            height_range=tuple(height_rng),
            min_gap=min_gap,
            boundary_margin=boundary_margin,
            max_tries_per_building=max_tries,
        )
    elif env == 'forest':
        csec = env_cfg.get('cylinders', {})
        cnt = sample_value(csec.get('count', 30), as_int=True)
        rr = csec.get('radius_range', [0.3, 0.8])
        hr = csec.get('height_range', [3.0, 8.0])
        gen.generate_random_cylinders(num_obstacles=cnt, radius_range=tuple(rr), height_range=tuple(hr), safe_margin=1.5)
    else:
        # custom: boxes + spheres + cylinders
        b = env_cfg.get('boxes', {})
        s = env_cfg.get('spheres', {})
        c = env_cfg.get('cylinders', {})
        if b:
            gen.generate_random_boxes(int(sample_value(b.get('count', 10), as_int=True)), tuple(b.get('size_range', [1.0, 4.0])), 2.0)
        if s:
            gen.generate_random_spheres(int(sample_value(s.get('count', 8), as_int=True)), tuple(s.get('radius_range', [0.6, 2.0])), 2.0)
        if c:
            gen.generate_random_cylinders(int(sample_value(c.get('count', 6), as_int=True)), tuple(c.get('radius_range', [0.4, 1.2])), tuple(c.get('height_range', [2.0, 6.0])), 1.5)


def _sample_free_point(gen: ObstacleMapGenerator, min_clearance: float, max_tries: int = 200) -> Optional[np.ndarray]:
    for _ in range(max_tries):
        p = np.random.rand(3) * gen.map_size
        # 将地图构建为中心坐标系：现有 C++ 规划器假设原点在中心，这里输出仍用 0..size 坐标，最终只要一致即可
        # 仅用 ESDF 判断清距
        d = gen.get_distance_at_world(p)
        if d >= min_clearance and gen.is_collision_free(p):
            return p
    return None


def _pick_start_goal(gen: ObstacleMapGenerator, min_dist: float, min_clearance: float, tries: int = 200) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    for _ in range(tries):
        s = _sample_free_point(gen, min_clearance, tries)
        g = _sample_free_point(gen, min_clearance, tries)
        if s is None or g is None:
            continue
        if np.linalg.norm(s - g) >= min_dist:
            return s, g
    return None


def _eval_poly_pos_vel_acc(coeff: np.ndarray, t: float):
    b = np.array([1, t, t**2, t**3, t**4, t**5], dtype=float)
    bv = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4], dtype=float)
    ba = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3], dtype=float)
    pos = coeff.T @ b
    vel = coeff.T @ bv
    acc = coeff.T @ ba
    return pos, vel, acc


def _sample_trajectory(coeffs: List[np.ndarray], intervals: List[float], dt: float):
    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    accelerations: List[np.ndarray] = []
    times: List[float] = []
    t_acc = 0.0
    for coeff, T in zip(coeffs, intervals):
        if T <= 0: continue
        ts = np.arange(0.0, float(T), float(dt))
        for t in ts:
            p, v, a = _eval_poly_pos_vel_acc(coeff, float(t))
            positions.append(p); velocities.append(v); accelerations.append(a)
            times.append(t_acc + float(t))
        p_end, v_end, a_end = _eval_poly_pos_vel_acc(coeff, float(T))
        positions.append(p_end); velocities.append(v_end); accelerations.append(a_end)
        times.append(t_acc + float(T))
        t_acc += float(T)
    return (
        np.asarray(positions, dtype=float),
        np.asarray(velocities, dtype=float),
        np.asarray(accelerations, dtype=float),
        np.asarray(times, dtype=float),
    )


def _terminal_attitude(positions: np.ndarray, velocities: np.ndarray):
    if len(positions) >= 2:
        dvec = positions[-1] - positions[-2]
        if np.linalg.norm(dvec) < 1e-6 and len(velocities) > 0:
            dvec = velocities[-1]
    else:
        dvec = np.array([1.0, 0.0, 0.0], dtype=float)
    vx, vy, vz = map(float, dvec)
    norm_xy = float(np.hypot(vx, vy))
    yaw = float(np.arctan2(vy, vx))
    pitch = float(np.arctan2(-vz, max(norm_xy, 1e-9)))  # 与 pipeline 保持一致
    roll = 0.0
    return {
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'final_tangent': [vx, vy, vz],
    }


def main():
    ap = argparse.ArgumentParser(description='批量生成数据集（单图多任务）')
    ap.add_argument('--config', type=str, default=None, help='配置 YAML/JSON 路径')
    ap.add_argument('--out-dir', type=str, default=None, help='输出目录（优先于配置）')
    ap.add_argument('--num-maps', type=int, default=None, help='生成地图数量（优先于配置）')
    ap.add_argument('--tasks-per-map', type=int, default=None, help='每张地图的任务数量（优先于配置）')
    ap.add_argument('--seed', type=int, default=None, help='随机种子（优先于配置）')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dataset_cfg = cfg.get('dataset', {})

    out_dir_path = Path(args.out_dir or dataset_cfg.get('output_dir', 'obstruct_env/datasets/out'))
    num_maps = args.num_maps if args.num_maps is not None else int(dataset_cfg.get('num_maps', 1))
    tasks_per_map = args.tasks_per_map if args.tasks_per_map is not None else int(dataset_cfg.get('tasks_per_map', 5))
    seed = args.seed if args.seed is not None else dataset_cfg.get('seed', None)
    if seed is not None:
        np.random.seed(int(seed)); random.seed(int(seed))

    out_dir, maps_dir, ds_dir = _ensure_dirs(out_dir_path)

    # 读取配置
    map_cfg = cfg.get('map', {})
    map_size = tuple(map_cfg.get('size', [20.0, 20.0, 10.0]))
    resolution = float(map_cfg.get('resolution', 0.2))

    planner_cfg = cfg.get('planner', {})
    stop_on_first = bool(planner_cfg.get('stop_on_first', True))
    planner_spec = {
        'step': planner_cfg.get('step', 1.0),
        'time_limit': planner_cfg.get('time_limit', 5.0),
        'max_iters': planner_cfg.get('max_iters', 5000),
        'inflate': planner_cfg.get('inflate', 0.0),
        'safety_margin': planner_cfg.get('safety_margin', 0.0),
        'safety_weight': planner_cfg.get('safety_weight', 1.0),
        'goal_tolerance': planner_cfg.get('goal_tolerance', 0.2),
    }

    dt = float(cfg.get('sampling', {}).get('dt', 0.05))

    task_cfg = cfg.get('task', {})
    min_dist_spec = task_cfg.get('min_distance', 5.0)
    min_clear_spec = task_cfg.get('min_clearance', 1.0)
    max_tries = int(task_cfg.get('max_tries', 200))
    sw_rng = task_cfg.get('preferences', {}).get('safety_weight_range', [0.5, 2.0])

    # 规划桥接器
    bridge = CppKinoRRTStarBridge(method='executable')

    index: List[Dict[str, Any]] = []

    for mi in range(1, num_maps + 1):
        map_id = f"map_{mi:05d}"
        print(f"\n=== 生成地图 {map_id} ===")
        gen = ObstacleMapGenerator(map_size=map_size, resolution=resolution)
        _gen_one_map(gen, map_cfg)
        esdf = gen.build_esdf().astype(np.float32)
        occ = gen.obstacle_map.astype(np.uint8)

        # 导出 base NPY（供 C++ 使用）
        base_path = maps_dir / map_id
        occ_path = Path(str(base_path) + '_occ.npy')
        esdf_path = Path(str(base_path) + '_esdf.npy')
        np.save(str(occ_path), occ)
        np.save(str(esdf_path), esdf)

        # 开始生成任务
        tasks: List[Dict[str, Any]] = []
        task_success = 0
        attempts = 0
        while task_success < tasks_per_map and attempts < tasks_per_map * 10:
            attempts += 1
            # 每个任务采样：约束与规划参数
            min_dist = float(sample_value(min_dist_spec, default=5.0))
            min_clearance = float(sample_value(min_clear_spec, default=1.0))

            step = float(sample_value(planner_spec['step'], default=1.0))
            time_limit = float(sample_value(planner_spec['time_limit'], default=5.0))
            max_iters = int(sample_value(planner_spec['max_iters'], as_int=True, default=5000))
            inflate = float(sample_value(planner_spec['inflate'], default=0.0))
            safety_margin = float(sample_value(planner_spec['safety_margin'], default=0.0))
            safety_weight_base = float(sample_value(planner_spec['safety_weight'], default=1.0))
            goal_tolerance = float(sample_value(planner_spec['goal_tolerance'], default=0.2))
            # 采样起终点
            pr = _pick_start_goal(gen, min_dist, min_clearance, tries=max_tries)
            if pr is None:
                continue
            sp, gp = pr
            # 转换到以地图中心为原点的坐标系（C++ 规划器坐标）
            shift = 0.5 * np.asarray(map_size, dtype=float)
            sp_c = sp - shift
            gp_c = gp - shift
            sv = np.zeros(3, dtype=float); sa = np.zeros(3, dtype=float)
            gv = np.zeros(3, dtype=float); ga = np.zeros(3, dtype=float)

            # 偏好：调整 safety_weight（叠乘 base 值）
            safety_weight = float(safety_weight_base * np.random.uniform(sw_rng[0], sw_rng[1]))

            # 调 C++ 规划器
            result = bridge.plan_with_map_base(
                start_pos=sp_c, start_vel=sv, start_acc=sa, goal_pos=gp_c,
                map_base=str(base_path), grid_res=resolution,
                inflate_radius=inflate, safety_margin=safety_margin, safety_weight=safety_weight,
                goal_vel=gv, goal_acc=ga,
                step=step, time_limit=time_limit, max_iters=max_iters,
                output_path=None,
            )
            if result is None or not result.get('success', False):
                continue

            coeffs = [np.asarray(c) for c in result['coefficients']]
            intervals = [float(t) for t in result['intervals']]
            positions, velocities, accelerations, times = _sample_trajectory(coeffs, intervals, dt)

            # 构建高层结果
            # 全局关键点与时间
            g_points: List[List[float]] = []
            g_times: List[float] = []
            t_cum = 0.0
            for i, (cf, T) in enumerate(zip(coeffs, intervals)):
                p0, _, _ = _eval_poly_pos_vel_acc(cf, 0.0)
                if i == 0:
                    g_points.append(p0.tolist()); g_times.append(float(t_cum))
                pT, _, _ = _eval_poly_pos_vel_acc(cf, float(T))
                t_cum += float(T)
                g_points.append(pT.tolist()); g_times.append(float(t_cum))

            local_segments = []
            for i in range(len(g_points) - 1):
                local_segments.append({
                    'segment': [g_points[i], g_points[i+1]],
                    'time_steps': [g_times[i], g_times[i+1]],
                })

            term_att = _terminal_attitude(positions, velocities)

            task_rec = {
                'start_state': np.array([sp_c[0], sp_c[1], sp_c[2], 0,0,0, 0,0,0], dtype=float),
                'goal_state':  np.array([gp_c[0], gp_c[1], gp_c[2], 0,0,0, 0,0,0], dtype=float),
                'constraints': {
                    'min_distance': min_dist,
                    'min_clearance': min_clearance,
                },
                'preferences': {
                    'safety_weight': safety_weight,
                },
                'planner_params': {
                    'step': step,
                    'time_limit': time_limit,
                    'max_iters': max_iters,
                    'stop_on_first': stop_on_first,
                    'inflate': inflate,
                    'safety_margin': safety_margin,
                    'safety_weight': safety_weight,
                    'goal_tolerance': goal_tolerance,
                },
                'result': {
                    'success': True,
                    'cost': float(result.get('cost', 0.0)),
                    'coefficients': coeffs,
                    'intervals': intervals,
                    'trajectory': {
                        'positions': positions,
                        'velocities': velocities,
                        'accelerations': accelerations,
                        'times': times,
                    },
                    'global_waypoints': g_points,
                    'time_steps': g_times,
                    'local_segments': local_segments,
                    'terminal_attitude': term_att,
                },
            }
            tasks.append(task_rec)
            task_success += 1

        # 写单图数据包
        pack = {
            'map_id': map_id,
            'obstacle_map': occ,
            'esdf': esdf,
            'resolution': float(resolution),
            'map_size': np.array(map_size, dtype=float),
            'frame_origin': 'center',
            'obstacles': gen.obstacles,
            'map_params': {
                'environment': map_cfg.get('environment', 'city'),
                'boxes': map_cfg.get('boxes', {}),
                'spheres': map_cfg.get('spheres', {}),
                'cylinders': map_cfg.get('cylinders', {}),
            },
            'planner_spec': planner_cfg,
            'sampling': {
                'dt': dt,
            },
            'task_spec': task_cfg,
            'tasks': tasks,
        }
        ds_file = ds_dir / f"{map_id}.npz"
        np.savez_compressed(ds_file, **pack)
        print(f"完成地图 {map_id}：共 {len(tasks)} 个任务，保存到 {ds_file}")

        index.append({
            'map_id': map_id,
            'file': str(ds_file),
            'grid_res': float(resolution),
            'map_size': list(map_size),
            'tasks': len(tasks),
        })

    # 汇总索引
    with open(out_dir / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n数据集完成：{len(index)} 张地图，索引写入 {out_dir / 'index.json'}")


if __name__ == '__main__':
    main()

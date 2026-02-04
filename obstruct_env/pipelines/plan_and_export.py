#!/usr/bin/env python3
"""
从 YAML 或命令行参数读取地图/起终点/规划参数，调用 C++ 规划器，输出完整的训练数据 NPZ。

输出内容：
- trajectory: positions, velocities, accelerations, times
- global_waypoints, time_steps
- local_segments: [{segment: [p_i, p_{i+1}], time_steps: [t_i, t_{i+1}]}]
- dynamics: velocity, acceleration, time_allocation
- path_cost: length_cost, safety_cost, smoothness_cost
- raw: coefficients, intervals, success, cost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 兼容直接运行
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


def _eval_poly_pos_vel_acc(coeff: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定单段多项式系数 (6,3) 与时间 t，返回 pos/vel/acc (3,)。
    约定：basis 与 bridge.sample_trajectory_from_coeffs 一致。
    """
    b = np.array([1, t, t**2, t**3, t**4, t**5], dtype=float)
    bv = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4], dtype=float)
    ba = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3], dtype=float)
    pos = coeff.T @ b
    vel = coeff.T @ bv
    acc = coeff.T @ ba
    return pos, vel, acc


def _solve_quintic_coeff(p0: np.ndarray, v0: np.ndarray, a0: np.ndarray,
                         pf: np.ndarray, vf: np.ndarray, af: np.ndarray,
                         T: float) -> np.ndarray:
    """解 5 次多项式系数，返回 (6,3) 矩阵，每列对应 x/y/z。
    与 standalone_trajectory_generator 中的形式一致。
    """
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3],
    ], dtype=float)
    coeff = np.zeros((6, 3), dtype=float)
    for d in range(3):
        b = np.array([p0[d], v0[d], a0[d], pf[d], vf[d], af[d]], dtype=float)
        coeff[:, d] = np.linalg.solve(A, b)
    return coeff


essential_fields = [
    "positions", "velocities", "accelerations", "times",
    "global_waypoints", "time_steps", "local_segments", "path_cost",
]


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        # 默认配置
        default_cfg = (_repo_root / "obstruct_env" / "config" / "planning.yaml").resolve()
        if not default_cfg.exists():
            raise FileNotFoundError("未提供 --config 且未找到默认配置 obstruct_env/config/planning.yaml")
        path = str(default_cfg)
        print(f"使用默认规划配置: {path}")
    cfg_path = Path(path)
    if cfg_path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("需要 PyYAML：请 pip install pyyaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif cfg_path.suffix.lower() == ".json":
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("仅支持 YAML/JSON 配置文件")


def compute_costs(
    positions: np.ndarray,
    accelerations: np.ndarray,
    times: np.ndarray,
    esdf_gen: Optional[ObstacleMapGenerator],
    safety_margin: float,
) -> Dict[str, float]:
    # 长度代价
    diffs = positions[1:] - positions[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    length_cost = float(np.sum(seg_len))

    # 平滑度代价（加速度平方积分近似）
    if len(times) >= 2:
        dt = float(np.mean(np.diff(times)))
    else:
        dt = 0.05
    smoothness_cost = float(np.sum(np.linalg.norm(accelerations, axis=1) ** 2) * dt)

    # 安全代价（若有 ESDF）
    safety_cost = 0.0
    if esdf_gen is not None and safety_margin > 0.0:
        vals = []
        for p in positions:
            d = esdf_gen.get_distance_at_world(np.asarray(p, dtype=float))
            vals.append(max(0.0, safety_margin - float(d)))
        safety_cost = float(np.sum(vals) * dt)

    return {
        "length_cost": length_cost,
        "safety_cost": safety_cost,
        "smoothness_cost": smoothness_cost,
    }


def _to_jsonable(obj):
    """递归将 numpy 类型转换为可 JSON 序列化的 Python 原生类型。"""
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    # 其他不可序列化对象转字符串
    return str(obj)


def main():
    ap = argparse.ArgumentParser(description="调用 C++ 规划器并导出完整训练数据 NPZ")
    ap.add_argument("--config", type=str, default=None, help="规划配置 YAML/JSON；未提供则读 obstruct_env/config/planning.yaml")
    ap.add_argument("--dt", type=float, default=None, help="覆盖配置中的采样 dt（秒）")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # 地图配置
    map_cfg = cfg.get("map", {})
    base = map_cfg.get("base")
    grid_res = float(map_cfg.get("grid_res", 0.2))
    if not base:
        raise ValueError("map.base 不能为空（请提供不带后缀的基路径）")

    # 起终点与规划参数
    sp = np.array(cfg.get("start", {}).get("position", [0, 0, 0]), dtype=float)
    sv = np.array(cfg.get("start", {}).get("velocity", [0, 0, 0]), dtype=float)
    sa = np.array(cfg.get("start", {}).get("acceleration", [0, 0, 0]), dtype=float)
    goal_cfg = cfg.get("goal", {})
    gp = np.array(goal_cfg.get("position", [1, 1, 1]), dtype=float)
    gv = np.array(goal_cfg.get("velocity", [0, 0, 0]), dtype=float)
    ga = np.array(goal_cfg.get("acceleration", [0, 0, 0]), dtype=float)

    planner_cfg = cfg.get("planner", {})
    inflate = float(planner_cfg.get("inflate", 0.0))
    safety_margin = float(planner_cfg.get("safety_margin", 0.0))
    safety_weight = float(planner_cfg.get("safety_weight", 0.0))

    # 采样步长
    dt = float(args.dt if args.dt is not None else cfg.get("sampling", {}).get("dt", 0.05))

    # 输出路径
    out_npz = cfg.get("output", {}).get("npz")
    out_json = cfg.get("output", {}).get("json")
    if out_npz is None:
        out_npz = str(Path(base).with_name(Path(base).name + "_traj.npz"))

    # 调用桥接器（map-base 模式）
    bridge = CppKinoRRTStarBridge(method="executable")
    result = bridge.plan_with_map_base(
        start_pos=sp, start_vel=sv, start_acc=sa, goal_pos=gp,
        map_base=str(base), grid_res=grid_res,
        inflate_radius=inflate, safety_margin=safety_margin, safety_weight=safety_weight,
        goal_vel=gv, goal_acc=ga,
        output_path=Path(out_json) if out_json else None,
    )
    if result is None or not result.get("success", False):
        print("规划失败或无结果")
        sys.exit(2)

    coeffs: List[np.ndarray] = result["coefficients"]
    intervals: List[float] = result["intervals"]

    # 采样轨迹（包含每段 [0, T)；最后补上终点）
    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    accelerations: List[np.ndarray] = []
    times: List[float] = []
    t_acc = 0.0
    for coeff, T in zip(coeffs, intervals):
        if T <= 0:
            continue
        ts = np.arange(0.0, T, dt)
        for t in ts:
            p, v, a = _eval_poly_pos_vel_acc(coeff, float(t))
            positions.append(p)
            velocities.append(v)
            accelerations.append(a)
            times.append(t_acc + float(t))
        # 补终点
        p_end, v_end, a_end = _eval_poly_pos_vel_acc(coeff, float(T))
        positions.append(p_end); velocities.append(v_end); accelerations.append(a_end)
        times.append(t_acc + float(T))
        t_acc += float(T)

    # 若末端速度不为零，附加“刹车段”将速度降为 0（基于最大加速度的估计）
    term_cfg = cfg.get("terminal", {})
    enforce_term = bool(term_cfg.get("enforce_zero_velocity", False))
    v_tol = float(term_cfg.get("velocity_tol", 0.05))
    a_max = float(term_cfg.get("max_acc", 2.0))
    t_min = float(term_cfg.get("min_time", 0.5))
    t_max = float(term_cfg.get("max_time", 2.0))
    if enforce_term and len(positions) > 0:
        v_last = velocities[-1]
        a_last = accelerations[-1]
        v_norm = float(np.linalg.norm(v_last))
        if v_norm > v_tol:
            # 估算停止距离与时间（常加速度近似）
            T_stop = np.clip(2.0 * v_norm / max(a_max, 1e-6), t_min, t_max)
            dir_v = (v_last / v_norm) if v_norm > 1e-9 else np.zeros(3)
            s_stop = (v_norm ** 2) / max(2.0 * a_max, 1e-6)
            p0 = positions[-1]
            v0 = v_last
            a0 = a_last
            pf = p0 + dir_v * s_stop
            vf = np.zeros(3)
            af = np.zeros(3)
            coeff_brake = _solve_quintic_coeff(p0, v0, a0, pf, vf, af, T_stop)

            # 刹车段安全增强：若有 ESDF，沿段采样碰撞检查与动态调整
            def _brake_segment_safe(_coeff: np.ndarray, _T: float, _esdf: Optional[ObstacleMapGenerator], _margin: float) -> bool:
                if _esdf is None:
                    return True
                dt_chk = max(0.02, dt)
                t = 0.0
                while t <= _T:
                    b = np.array([1, t, t**2, t**3, t**4, t**5], dtype=float)
                    pos = _coeff.T @ b
                    d = _esdf.get_distance_at_world(np.asarray(pos, dtype=float))
                    if float(d) < 0.0:  # 碰撞
                        return False
                    if _margin > 0.0 and float(d) < _margin * 0.5:  # 过近，留冗余
                        return False
                    t += dt_chk
                return True

            if 'esdf_gen' in locals() and esdf_gen is not None:
                tries = 0
                safe = _brake_segment_safe(coeff_brake, T_stop, esdf_gen, safety_margin)
                while not safe and tries < 5:
                    tries += 1
                    # 延长时间放缓段末动态，重算系数
                    T_stop = min(t_max, T_stop * 1.25)
                    coeff_brake = _solve_quintic_coeff(p0, v0, a0, pf, vf, af, T_stop)
                    safe = _brake_segment_safe(coeff_brake, T_stop, esdf_gen, safety_margin)
            coeffs.append(coeff_brake)
            intervals.append(float(T_stop))
            tsb = np.arange(0.0, T_stop, dt)
            for t in tsb:
                p, v, a = _eval_poly_pos_vel_acc(coeff_brake, float(t))
                positions.append(p)
                velocities.append(v)
                accelerations.append(a)
                times.append(t_acc + float(t))
            p_end2, v_end2, a_end2 = _eval_poly_pos_vel_acc(coeff_brake, float(T_stop))
            positions.append(p_end2); velocities.append(v_end2); accelerations.append(a_end2)
            times.append(t_acc + float(T_stop))
            t_acc += float(T_stop)

    positions_np = np.asarray(positions, dtype=float)
    velocities_np = np.asarray(velocities, dtype=float)
    accelerations_np = np.asarray(accelerations, dtype=float)
    times_np = np.asarray(times, dtype=float)

    # 构造全局关键点（各段端点）与时间
    g_points: List[List[float]] = []
    g_times: List[float] = []
    t_cum = 0.0
    for i, (coeff, T) in enumerate(zip(coeffs, intervals)):
        p0, _, _ = _eval_poly_pos_vel_acc(coeff, 0.0)
        if i == 0:
            g_points.append(p0.tolist())
            g_times.append(float(t_cum))
        pT, _, _ = _eval_poly_pos_vel_acc(coeff, float(T))
        t_cum += float(T)
        g_points.append(pT.tolist())
        g_times.append(float(t_cum))

    # 局部段
    local_segments: List[Dict[str, Any]] = []
    for i in range(len(g_points) - 1):
        local_segments.append({
            "segment": [g_points[i], g_points[i+1]],
            "time_steps": [g_times[i], g_times[i+1]],
        })

    # 安全代价辅助（如有 ESDF）
    esdf_gen: Optional[ObstacleMapGenerator] = None
    esdf_path = Path(str(base) + "_esdf.npy")
    occ_path = Path(str(base) + "_occ.npy")
    if esdf_path.exists() and occ_path.exists():
        occ = np.load(str(occ_path))
        esdf = np.load(str(esdf_path)).astype(np.float32)
        grid_size = np.array(occ.shape, dtype=int)
        map_size = grid_size * float(grid_res)
        esdf_gen = ObstacleMapGenerator(map_size=tuple(map_size), resolution=float(grid_res))
        esdf_gen.obstacle_map = occ.astype(np.uint8)
        esdf_gen.esdf = esdf

    # 代价计算
    path_cost = compute_costs(positions_np, accelerations_np, times_np, esdf_gen, safety_margin)

    # 打包结果
    data = {
        "trajectory": {
            "positions": positions_np,
            "velocities": velocities_np,
            "accelerations": accelerations_np,
            "times": times_np,
        },
        "global_waypoints": g_points,
        "time_steps": g_times,
        "local_segments": local_segments,
        "dynamics": {
            "velocity": velocities_np,
            "acceleration": accelerations_np,
            "time_allocation": times_np,
        },
        "path_cost": path_cost,
        "raw": {
            "coefficients": [np.asarray(c) for c in coeffs],
            "intervals": np.asarray(intervals, dtype=float),
            "success": bool(result.get("success", True)),
            "cost": float(result.get("cost", 0.0)),
        },
    }

    # 附加：地图与起终点元信息，便于后续验证/训练
    occ_for_meta = None
    occ_path = Path(str(base) + "_occ.npy")
    if occ_path.exists():
        occ_for_meta = np.load(str(occ_path)).astype(np.uint8)
        grid_size = np.array(occ_for_meta.shape, dtype=int)
        map_size = grid_size * float(grid_res)
        data["obstacle_map"] = occ_for_meta
        data["map_info"] = {
            "grid_res": float(grid_res),
            "grid_size": grid_size.astype(int),
            "map_size": map_size.astype(float),
            "base": str(base),
        }
    # 起终点与末端姿态估算
    data["start_state"] = np.array([sp[0], sp[1], sp[2], 0.0, 0.0, 0.0], dtype=float)
    # 使用全局关键点最后一个作为 goal（与实际多段末端一致）
    goal_last = np.array(g_points[-1], dtype=float) if len(g_points) > 0 else gp

    # 末端姿态：取末端切向方向（最后两个点或末端速度）
    if len(positions_np) >= 2:
        dvec = positions_np[-1] - positions_np[-2]
        if np.linalg.norm(dvec) < 1e-6 and len(velocities_np) > 0:
            dvec = velocities_np[-1]
    else:
        dvec = np.array([1.0, 0.0, 0.0], dtype=float)
    vx, vy, vz = float(dvec[0]), float(dvec[1]), float(dvec[2])
    norm_xy = float(np.hypot(vx, vy))
    yaw = float(np.arctan2(vy, vx))
    pitch = float(np.arctan2(-vz, max(norm_xy, 1e-9)))  # 约定：向上为负 pitch，可按需要调整
    roll = 0.0

    data["goal_state"] = np.array([goal_last[0], goal_last[1], goal_last[2], yaw, pitch, roll], dtype=float)
    data["terminal_attitude"] = {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "final_tangent": [vx, vy, vz],
    }

    out_npz_path = Path(out_npz)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz_path, **data)

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(data), f, ensure_ascii=False, indent=2)

    print(f"规划完成，已保存：{out_npz_path}")


if __name__ == "__main__":
    main()

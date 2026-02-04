# 数据集批量生成工具

本工具从“地图生成 → ESDF 构建 → 多任务规划 → 验证与打包”一条龙生成数据集。

- 每个地图保存为一个 .npz，包含：obstacle_map、esdf、分辨率/尺寸、障碍物列表、多个任务（起终点/偏好/轨迹/系数/代价等）。
- 同时在 maps/ 下导出 `<base>_occ.npy` 与 `<base>_esdf.npy` 以便 C++ 规划器快速复用。

## 快速开始

1) 使用默认配置小规模试跑：

```bash
python3 obstruct_env/datasets/generate_dataset.py --config obstruct_env/datasets/dataset_config.yaml
```

> 说明：命令行参数可覆盖配置文件。例如 `--num-maps 4` 会替换配置中的 `dataset.num_maps`。

2) 产物结构（示例）：

```
obstruct_env/datasets/out/
  maps/
    map_00001_occ.npy
    map_00001_esdf.npy
  dataset/
    map_00001.npz   # 包含地图 + 3 个任务的完整数据
  index.json        # 汇总索引（可选）
```

## 配置说明（dataset_config.yaml）

```
dataset:             # 数据集级别参数
  output_dir: ...    # 输出根目录（含 maps/ 与 dataset/）
  num_maps: 1        # 要生成的地图张数（map_00001, map_00002, ...）
  tasks_per_map: 5   # 每张地图需要成功写入的数据条数/路径数量
  seed: 2025         # 可选，固定随机种子便于复现

map:                 # 地图与障碍物生成
  size: [...]        # 物理尺寸（米）
  resolution: 0.2    # 栅格分辨率
  environment: city  # city|buildings|forest|custom
  boxes/spheres/...  # 各类障碍的数量范围与尺寸范围

planner:             # 规划器参数，可为常数、范围或列表
  step: [0.6, 1.4]   # 扩展步长（米）
  ...                # time_limit, max_iters, inflate, safety_margin, ...

sampling:
  dt: 0.05           # 轨迹离散采样间隔（秒）

task:                # 起点/终点采样约束与偏好
  min_distance: ...  # 起终点最小距离
  min_clearance: ... # 点到障碍的最小清距
  preferences: ...   # per-task safety_weight 扰动等
```

每个字段在 `dataset_config.yaml` 中都配有中文注释，可直接修改。

## 输出 schema（map_00001.npz）
- obstacle_map: uint8 (nx,ny,nz)
- esdf: float32 (nx,ny,nz)
- resolution: float
- map_size: float[3]
- obstacles: list[dict]
- tasks: list[dict]
  - start_state: [x,y,z,vx,vy,vz,ax,ay,az]
  - goal_state:  同上
  - preferences: { safety_weight, notes ... }
  - constraints: { min_distance, min_clearance }
  - planner_params: 实际调用参数记录（即生成该轨迹时使用的规划器参数）
  - result:
    - success: bool
    - cost: float
    - coefficients: list[(6,3)]
    - intervals: list[float]
    - trajectory: { positions, velocities, accelerations, times }
    - global_waypoints, time_steps, local_segments
    - terminal_attitude: { yaw,pitch,roll,final_tangent }

额外元数据：
- planner_spec: 保存配置中的 planner 字段（便于追溯参数分布）
- sampling: { dt }
- task_spec: 原始 task 配置

注意：如需最小体积存储，可仅保留系数 + 采样步长，运行时再重建采样轨迹。

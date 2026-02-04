
用于生成无人机轨迹规划的训练数据集，包含 **3D 障碍物地图** 和 **符合动力学约束的轨迹**。

## 📋 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    数据集生成流程                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  1. obstruct_env - 环境与地图生成         │
        │     ├─ 生成 3D 障碍物地图（占据栅格）       │
        │     ├─ 计算 ESDF（距离场）                │
        │     └─ 采样起点与终点                     │
        └────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  2. cpp_bridge - Python-C++ 桥接        │
        │     └─ 将地图和任务传递给 C++ 规划器      │
        └────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  3. cpp/kinodyn_rrt_star - 轨迹规划    │
        │     ├─ Kinodynamic RRT* 算法            │
        │     ├─ 生成多项式轨迹系数                 │
        │     └─ 输出轨迹采样点与元数据             │
        └────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │  4. 打包输出 - NPZ 格式数据集            │
        │     ├─ 地图 + ESDF                      │
        │     ├─ 多个任务（起点、终点、偏好）        │
        │     └─ 轨迹（系数、采样点、代价）          │
        └────────────────────────────────────────┘
```

---

## 📁 核心目录结构

### `obstruct_env/` - 环境生成模块

```
obstruct_env/
├── core/
│   ├── obstacle_map.py        # 3D 占据栅格地图生成器
│   └── esdf.py                # 欧几里得符号距离场计算
├── generators/
│   ├── advanced_obstacle_generator.py  # 高级障碍物生成
│   ├── environment_builder.py          # 环境构建器
│   └── generate_map.py                 # 地图生成工具
├── datasets/
│   ├── generate_dataset.py    # 🎯 主数据集生成脚本
│   ├── dataset_config.yaml    # 配置文件
│   └── README.md              # 使用说明
└── visualization/             # 可视化工具
```

**主要功能：**
- 生成多种类型的障碍物（盒子、球体、圆柱、椭球）
- 支持预设环境（城市、森林、建筑群）
- 计算 ESDF 用于软约束规划
- 智能采样起点和终点（保证可达性和最小距离）

### `cpp/kinodyn_rrt_star/` - C++ 轨迹规划器

```
cpp/kinodyn_rrt_star/
├── include/
│   └── kinodyn_rrt_star/
│       └── kinodyn_rrt_star.h    # 公共 API
├── src/
│   ├── kinodyn_rrt_star.cpp      # 规划器实现
│   └── kdtree.cpp                # KD 树实现
├── app/
│   └── main.cpp                  # 命令行入口
├── build/                        # 编译输出目录
└── CMakeLists.txt                # 构建配置
```

**主要功能：**
- Kinodynamic RRT* 采样规划
- 考虑速度、加速度约束
- 生成平滑的多项式轨迹
- 支持 ESDF 软安全代价优化

### 桥接文件

- **`cpp_bridge.py`** - Python 调用 C++ 规划器
  - 支持可执行文件模式（通过 JSON 通信）
  - 支持 pybind11 模式（直接调用，更快）
  
- **`standalone_trajectory_generator.py`** - 纯 Python 备用实现
  - 不依赖 C++ 编译
  - 功能相对简化，速度较慢

---

## 🚀 快速开始

### 1. 环境准备

#### 安装 Python 依赖
```bash
pip install -r requirements.txt
```

主要依赖：
- `numpy` - 数组计算
- `scipy` - ESDF 距离变换
- `pyyaml` - 配置文件解析
- `matplotlib` - 可视化（可选）

#### 编译 C++ 规划器
```bash
cd cpp/kinodyn_rrt_star
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

依赖：
- CMake >= 3.10
- C++14 编译器（GCC/Clang）
- Eigen3 库（`sudo apt install libeigen3-dev`）

验证编译成功：
```bash
./bin/kino_rrt_planner --help
```

### 2. 生成数据集

#### 使用默认配置
```bash
python3 obstruct_env/datasets/generate_dataset.py \
    --config obstruct_env/datasets/dataset_config.yaml
```

#### 自定义参数
```bash
python3 obstruct_env/datasets/generate_dataset.py \
    --config obstruct_env/datasets/dataset_config.yaml \
    --num-maps 10 \          # 生成 10 张地图
    --tasks-per-map 5 \      # 每张地图 5 条轨迹
    --seed 42                # 固定随机种子
```

#### 输出结构
```
obstruct_env/datasets/out/
├── maps/
│   ├── map_00001_occ.npy    # 占据栅格
│   ├── map_00001_esdf.npy   # 距离场
│   ├── map_00002_occ.npy
│   └── ...
├── dataset/
│   ├── map_00001.npz        # 完整数据包（地图+轨迹）
│   ├── map_00002.npz
│   └── ...
└── index.json               # 数据集索引（可选）
```

---

## ⚙️ 配置说明

配置文件位置：`obstruct_env/datasets/dataset_config.yaml`

### 数据集配置
```yaml
dataset:
  output_dir: obstruct_env/datasets/out  # 输出目录
  num_maps: 10                           # 地图数量
  tasks_per_map: 5                       # 每张地图的轨迹数量
  seed: 2025                             # 随机种子（可选）
```

### 地图配置
```yaml
map:
  size: [20.0, 20.0, 10.0]    # 地图尺寸 (x, y, z) 米
  resolution: 0.2             # 栅格分辨率 米/格
  environment: city           # 环境类型：city|buildings|forest|custom
  
  # 障碍物配置（支持范围或固定值）
  boxes:
    count: [10, 20]           # 盒子数量范围
    size_range: [2.0, 5.0]    # 尺寸范围
  
  buildings:
    count: [15, 30]
    footprint_range: [2.0, 6.0]
    height_range: [3.0, 10.0]
    min_gap: 1.0              # 建筑间距
  
  spheres:
    count: [6, 12]
    radius_range: [0.6, 2.0]
  
  cylinders:
    count: [5, 10]
    radius_range: [0.4, 1.2]
    height_range: [2.0, 6.0]
```

### 规划器配置
```yaml
planner:
  step: [0.6, 1.4]              # 采样步长范围（米）
  time_limit: [3.0, 8.0]        # 规划时间限制（秒）
  max_iters: [3000, 8000]       # 最大迭代次数
  stop_on_first: true           # 找到首个解即停止
  
  # 安全参数
  inflate: [0.3, 0.8]           # 硬膨胀半径（米）
  safety_margin: [0.4, 1.0]     # 软安全距离（米）
  safety_weight: [0.5, 2.0]     # 安全代价权重
  
  goal_tolerance: [0.01, 0.001] # 目标容差（米）
```

### 任务配置
```yaml
task:
  min_distance: [10.0, 20.0]    # 起终点最小距离（米）
  min_clearance: [0.8, 1.5]     # 采样点到障碍物最小距离（米）
  max_tries: 200                # 采样重试次数
  
  preferences:
    safety_weight_range: [0.5, 2.0]  # 每个任务的安全偏好扰动
```

### 采样配置
```yaml
sampling:
  dt: 0.05    # 轨迹离散采样时间步长（秒）
```

---

## 📦 输出数据格式

每个 `.npz` 文件包含以下字段：

### 地图相关
```python
data = np.load('map_00001.npz', allow_pickle=True)

# 占据栅格 (nx, ny, nz)，0=自由空间，1=障碍物
obstacle_map = data['obstacle_map']  # uint8

# 欧几里得符号距离场 (nx, ny, nz)
esdf = data['esdf']  # float32

# 地图元数据
resolution = data['resolution']  # float, 如 0.2
map_size = data['map_size']      # [x, y, z], 如 [20, 20, 10]

# 障碍物列表
obstacles = data['obstacles']    # list of dict
# 每个障碍物: {type, center, size/radius/height, ...}
```

### 任务与轨迹
```python
tasks = data['tasks']  # list of dict

for task in tasks:
    # 起点和终点状态 [x, y, z, vx, vy, vz, ax, ay, az]
    start_state = task['start_state']  # shape (9,)
    goal_state = task['goal_state']    # shape (9,)
    
    # 任务偏好
    preferences = task['preferences']
    # {'safety_weight': 1.2, 'notes': 'high_safety', ...}
    
    # 约束条件
    constraints = task['constraints']
    # {'min_distance': 15.0, 'min_clearance': 1.0}
    
    # 规划器参数（实际使用值）
    planner_params = task['planner_params']
    # {'step': 1.0, 'time_limit': 5.0, 'inflate': 0.5, ...}
    
    # 规划结果
    result = task['result']
    
    if result['success']:
        # 轨迹代价
        cost = result['cost']
        
        # 多项式系数 (M, 6, 3)
        # M 段，6 阶多项式，3 个维度 (x, y, z)
        coefficients = result['coefficients']
        
        # 时间间隔 (M,)
        intervals = result['intervals']
        
        # 离散采样轨迹
        trajectory = result['trajectory']
        positions = trajectory['positions']      # (N, 3)
        velocities = trajectory['velocities']    # (N, 3)
        accelerations = trajectory['accelerations']  # (N, 3)
        times = trajectory['times']              # (N,)
        
        # 全局路径点
        global_waypoints = result['global_waypoints']  # (M+1, 3)
        
        # 终端姿态
        terminal_attitude = result['terminal_attitude']
        # {'yaw': ..., 'pitch': ..., 'roll': ..., 'final_tangent': [...]}
```

### 配置存档
```python
# 保存生成时的配置，便于追溯
planner_spec = data['planner_spec']  # dict
sampling = data['sampling']          # {'dt': 0.05}
task_spec = data['task_spec']        # dict
```

---

## 🔧 高级使用

### 1. 仅生成地图（不规划轨迹）

修改 `generate_dataset.py` 或自定义脚本：

```python
from obstruct_env.core.obstacle_map import ObstacleMapGenerator
from obstruct_env.core.esdf import compute_esdf

# 创建地图生成器
gen = ObstacleMapGenerator(map_size=(20, 20, 10), resolution=0.2)

# 添加障碍物
gen.add_random_obstacles(
    num_boxes=15,
    num_spheres=8,
    box_size_range=(2.0, 5.0),
    sphere_radius_range=(0.6, 2.0)
)

# 获取占据栅格
obstacle_map = gen.obstacle_map

# 计算 ESDF
esdf = compute_esdf(obstacle_map, resolution=0.2)

# 保存
np.savez('custom_map.npz', 
         obstacle_map=obstacle_map, 
         esdf=esdf,
         resolution=0.2,
         map_size=[20, 20, 10])
```

### 2. 单独调用规划器

```python
from cpp_bridge import CppKinoRRTStarBridge
import numpy as np

# 创建桥接器（需要先编译 C++）
bridge = CppKinoRRTStarBridge(method="executable")

# 加载地图
data = np.load('map_00001.npz', allow_pickle=True)
obstacle_map = data['obstacle_map']
esdf = data['esdf']
resolution = float(data['resolution'])
map_size = data['map_size']

# 定义起点和终点
start_pos = np.array([1.0, 1.0, 2.0])
start_vel = np.array([0.0, 0.0, 0.0])
start_acc = np.array([0.0, 0.0, 0.0])
goal_pos = np.array([18.0, 18.0, 8.0])

# 调用规划
result = bridge.plan(
    start_pos=start_pos,
    start_vel=start_vel,
    start_acc=start_acc,
    goal_pos=goal_pos,
    obstacle_map=obstacle_map,
    esdf=esdf,
    resolution=resolution,
    map_size=map_size,
    inflate_radius=0.5,
    safety_margin=0.8,
    safety_weight=1.0
)

if result and result['success']:
    print(f"规划成功！代价: {result['cost']}")
    trajectory = result['trajectory']
    # trajectory 包含 positions, velocities, accelerations, times
```

### 3. 可视化

```python
from obstruct_env.visualization import plot_trajectory_3d
import matplotlib.pyplot as plt

data = np.load('map_00001.npz', allow_pickle=True)
task = data['tasks'][0]

if task['result']['success']:
    trajectory = task['result']['trajectory']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    positions = trajectory['positions']
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, label='Trajectory')
    
    # 绘制起终点
    start = task['start_state'][:3]
    goal = task['goal_state'][:3]
    ax.scatter(*start, c='green', s=100, marker='o', label='Start')
    ax.scatter(*goal, c='red', s=100, marker='*', label='Goal')
    
    ax.legend()
    plt.show()
```

---

## 🎯 典型工作流

### 科研/实验流程

1. **配置实验参数**
   ```bash
   # 编辑配置文件
   vim obstruct_env/datasets/dataset_config.yaml
   ```

2. **生成训练集**
   ```bash
   python3 obstruct_env/datasets/generate_dataset.py \
       --config obstruct_env/datasets/dataset_config.yaml \
       --num-maps 100 --tasks-per-map 10 \
       --out-dir data/train --seed 42
   ```

3. **生成验证集**
   ```bash
   python3 obstruct_env/datasets/generate_dataset.py \
       --config obstruct_env/datasets/dataset_config.yaml \
       --num-maps 20 --tasks-per-map 10 \
       --out-dir data/val --seed 123
   ```

4. **生成测试集**（不同分布）
   ```bash
   # 可修改配置创建更困难的测试场景
   python3 obstruct_env/datasets/generate_dataset.py \
       --config obstruct_env/datasets/dataset_config_hard.yaml \
       --num-maps 50 --tasks-per-map 5 \
       --out-dir data/test --seed 456
   ```

5. **验证数据质量**
   ```bash
   python3 verify_trajectory_data.py
   python3 verify_data_flow.py
   ```

6. **训练模型**
   ```bash
   cd CHDM
   python3 train.py --config config/train_config.yaml
   ```

---

## 📊 性能参数

### 地图生成速度
- 简单地图（10-20 障碍物）：~0.5 秒/张
- 复杂地图（50+ 障碍物）：~2 秒/张
- ESDF 计算：~0.3 秒（20×20×10 米，0.2 米分辨率）

### 轨迹规划速度
- 简单场景：1-3 秒/条
- 复杂场景：5-10 秒/条
- 失败重试：自动重新采样起终点

### 数据集规模
- 单个 .npz 文件：~500 KB - 2 MB（取决于地图复杂度和轨迹数量）
- 100 张地图 × 10 条轨迹：~50-150 MB

---

## 🐛 常见问题

### Q1: C++ 编译失败
**A:** 检查依赖项
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libeigen3-dev

# 指定 Eigen 路径
cmake -DEigen3_DIR=/usr/share/eigen3/cmake ..
```

### Q2: 规划经常失败
**A:** 调整配置参数
- 增加 `planner.time_limit` 和 `planner.max_iters`
- 减小 `planner.inflate` 或 `task.min_clearance`
- 增加 `task.max_tries` 采样重试次数
- 降低障碍物密度

### Q3: 内存占用过大
**A:** 优化数据集
- 减小地图尺寸或增大分辨率（降低栅格数）
- 仅保存轨迹系数，不保存完整采样点
- 分批生成和处理数据

### Q4: 如何加速生成？
**A:** 并行化
- 使用 Python `multiprocessing` 并行生成多张地图
- 使用 pybind11 模式而非可执行文件模式
- 预编译 C++ 为 Release 模式（`cmake -DCMAKE_BUILD_TYPE=Release ..`）

### Q5: 轨迹不平滑
**A:** 调整规划参数
- 减小 `planner.step`（更细的采样）
- 增加 `safety_weight`（更平滑但可能绕远）
- 检查速度和加速度约束是否合理

---

## 📚 相关文档

- [READY_TO_TRAIN.md](READY_TO_TRAIN.md) - 模型训练指南
- [obstruct_env/datasets/README.md](obstruct_env/datasets/README.md) - 数据集工具详细说明
- [cpp/kinodyn_rrt_star/README.md](cpp/kinodyn_rrt_star/README.md) - C++ 规划器文档
- [CHDM/models/CHDMX_ARCHITECTURE.md](CHDM/models/CHDMX_ARCHITECTURE.md) - 模型架构说明

---

## 🤝 贡献

如需修改或扩展功能：

1. **添加新的障碍物类型**：编辑 `obstruct_env/core/obstacle_map.py`
2. **添加新的环境预设**：编辑 `obstruct_env/generators/environment_builder.py`
3. **修改规划器参数**：编辑 `cpp/kinodyn_rrt_star/src/kinodyn_rrt_star.cpp`
4. **自定义数据格式**：编辑 `obstruct_env/datasets/generate_dataset.py`

---

## 📝 许可证

请参考项目根目录的 LICENSE 文件。

---

**最后更新**: 2026年2月4日

# Kinodynamic RRT* 轨迹生成器

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**无需 ROS 的独立无人机 3D 轨迹生成工具**

基于 Kinodynamic RRT* 算法，生成考虑动力学约束的无人机轨迹数据，可直接用于 gym-pybullet-drones 或机器学习训练。

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install numpy matplotlib scipy
```

### 2. 使用虚拟环境模块（新增）
```python
from 虚拟环境 import EnvironmentBuilder
import numpy as np

# 快速构建森林场景
builder = EnvironmentBuilder(map_size=(20.0, 20.0, 10.0), resolution=0.2)
generator = builder.quick_build('forest', num_trees=40)

# 查看环境摘要
generator.print_summary()

# 保存场景
builder.save_scene('my_forest')
```

完整示例：
```bash
python3 虚拟环境/demo_full_example.py
```

### 3. 生成轨迹数据
```bash
python3 standalone_trajectory_generator.py
```

### 4. 验证数据
```bash
python3 verify_trajectory_data.py
```

### 5. 查看结果
```bash
ls -lh kino_rrt_trajectories/
```

---

## 📁 项目结构

```
kinodynamic_rrt_star/
├── README.md                              # 项目说明（本文件）
├── LICENSE                                # 许可证
│
├── standalone_trajectory_generator.py     # 🌟 主轨迹生成器
├── verify_trajectory_data.py              # 🔍 数据验证工具
├── cpp_bridge.py                          # 🔗 C++ 桥接接口
│
├── obstruct_env/                               # 🎯 环境构建模块（新增）
│   ├── README.md                          # 详细文档
│   ├── QUICKSTART.md                      # 快速开始指南
│   ├── obstacle_map_generator.py          # 障碍物地图生成器
│   ├── environment_builder.py             # 环境构建器（预定义场景）
│   ├── environment_visualizer.py          # 环境可视化工具
│   ├── demo_full_example.py               # 完整示例
│   └── examples/                          # 示例输出
│       ├── scenes/                        # 预定义场景
│       └── visualizations/                # 可视化图像
│
├── src/                                   # C++ 源代码
│   └── my_simple_planner/
│       ├── README.md                      # C++ 算法说明
│       ├── include/                       # 头文件
│       │   ├── kinodyn_rrt_star/         # RRT* 算法
│       │   └── kdtree/                    # KD-Tree
│       └── src/                           # 源文件
│           ├── kinodyn_rrt_star/
│           └── kdtree/
│
├── kino_rrt_trajectories/                 # 📊 生成的轨迹数据
│   ├── case_00000.npz
│   ├── case_00001.npz
│   └── ...
│
├── docs/                                  # 📚 文档
│   ├── README.md                          # 详细使用指南
│   ├── SUMMARY.md                         # 项目总结
│   └── CLEANUP_REPORT.md                  # 清理报告
│
├── examples/                              # 📝 使用示例
│   └── (即将添加)
│
└── build/                                 # 🔨 编译输出
    └── (编译产物)
```

---

## 🎯 核心功能

### ✅ 独立运行
- **无需 ROS** - 纯 Python 实现
- **无需仿真** - 直接生成数据
- **开箱即用** - 安装即可使用

### ✅ 标准数据格式
```python
{
    # 元数据
    "episode_id": "case_00003",
    "planner_type": "kinodynamic_rrt_star",
    
    # 3D 环境地图
    "global_map": {
        "occupancy_grid": np.array(shape=(100, 100, 50)),
        "resolution": 0.2,
        "size": [20.0, 20.0, 10.0]
    },
    
    # 任务定义
    "start_state": [x, y, z, yaw, pitch, roll],
    "goal_state": [x, y, z, yaw, pitch, roll],
    
    # 完整轨迹（164个点，8.15秒）
    "trajectory": {
        "positions": (164, 3),        # [x, y, z]
        "velocities": (164, 3),       # [vx, vy, vz]
        "accelerations": (164, 3),    # [ax, ay, az]
        "orientations": (164, 3),     # [yaw, pitch, roll]
        "timestamps": (164,)          # 时间戳
    },
    
    # 质量指标
    "rewards": {
        "total_length": 16.29,        # 路径长度 (m)
        "safety_margin": 10.0,        # 安全距离 (m)
        "smoothness": 0.37,           # 平滑度
        "execution_time": 8.20        # 执行时间 (s)
    }
}
```

### ✅ 高质量轨迹
- 基于 **Kinodynamic RRT*** 算法
- 考虑**速度和加速度**约束
- **5次多项式**平滑轨迹
- 包含完整的**状态信息**

### ✅ 虚拟环境构建（新增）
- **多种障碍物类型** - 盒子、球体、圆柱体、椭球体
- **预定义场景** - 森林、城市、迷宫、走廊等 8 种场景
- **3D 可视化** - 2D 切片和 3D 障碍物视图
- **与规划器集成** - 直接生成可用的占据栅格地图

快速开始：
```python
from 虚拟环境 import EnvironmentBuilder

builder = EnvironmentBuilder()
generator = builder.quick_build('forest', num_trees=40)
builder.save_scene('my_forest')
```

详细文档：[虚拟环境/README.md](./虚拟环境/README.md) | [快速开始](./虚拟环境/QUICKSTART.md)

---

## 📊 使用示例

### 示例 1: 使用虚拟环境模块

```python
from 虚拟环境 import ObstacleMapGenerator, EnvironmentVisualizer
import numpy as np

# 创建地图生成器
generator = ObstacleMapGenerator(
    map_size=(20.0, 20.0, 10.0),
    resolution=0.2
)

# 添加障碍物
generator.add_box_obstacle(
    center=np.array([10.0, 10.0, 2.5]),
    size=np.array([2.0, 2.0, 5.0])
)

generator.add_sphere_obstacle(
    center=np.array([5.0, 5.0, 3.0]),
    radius=1.5
)

# 保存地图
generator.save("my_map.npz")

# 可视化
visualizer = EnvironmentVisualizer(generator)
visualizer.visualize_3d_obstacles()
```

### 示例 2: 加载轨迹数据

```python
import numpy as np

# 加载轨迹数据
data = np.load('kino_rrt_trajectories/case_00003.npz', allow_pickle=True)

# 访问轨迹
positions = data['positions']        # (164, 3) - 位置序列
velocities = data['velocities']      # (164, 3) - 速度序列
timestamps = data['timestamps']      # (164,) - 时间戳

# 访问元数据
metadata = data['metadata'].item()
print(f"轨迹长度: {metadata['rewards']['total_length']:.2f} m")
print(f"执行时间: {metadata['rewards']['execution_time']:.2f} s")
```

### 在 gym-pybullet-drones 中使用
```python
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# 加载轨迹
data = np.load('kino_rrt_trajectories/case_00003.npz', allow_pickle=True)
positions = data['positions']
velocities = data['velocities']

# 创建环境和控制器
env = CtrlAviary(num_drones=1, gui=True)
ctrl = DSLPIDControl(DroneModel.CF2X)

# 跟踪轨迹
obs = env.reset()
for pos, vel in zip(positions, velocities):
    action, _, _ = ctrl.computeControl(
        control_timestep=0.05,
        cur_pos=obs[0][0:3],
        cur_quat=obs[0][3:7],
        cur_vel=obs[0][10:13],
        cur_ang_vel=obs[0][13:16],
        target_pos=pos,
        target_vel=vel
    )
    obs, _, _, _ = env.step(action.reshape(1, 4))
    env.render()
```

---

## 🔧 高级功能

### 自定义参数
```python
from standalone_trajectory_generator import TrajectoryDatasetGenerator

# 创建生成器
generator = TrajectoryDatasetGenerator(output_dir="./my_trajectories")

# 自定义地图和参数
generator.planner.map_size = np.array([30.0, 30.0, 15.0])  # 更大的地图
generator.planner.max_vel = 5.0                             # 更高速度
generator.planner.max_acc = 5.0                             # 更大加速度

# 生成数据集
generator.generate_dataset(num_episodes=1000)
```

### 集成 C++ RRT* 实现
```bash
# 使用 pybind11 编译 Python 绑定
pip install pybind11

c++ -O3 -shared -std=c++14 -fPIC \
    $(python3 -m pybind11 --includes) \
    python_bindings.cpp \
    -o kinodyn_rrt_star_py.so \
    -I src/my_simple_planner/include

# 在 Python 中使用
from cpp_bridge import CppKinoRRTStarBridge
bridge = CppKinoRRTStarBridge(method='pybind11')
```

---

## 📈 性能指标

| 指标 | 典型值 |
|------|--------|
| 轨迹点数 | 100-200 点 |
| 持续时间 | 5-10 秒 |
| 路径长度 | 10-30 米 |
| 最大速度 | 2-4 m/s |
| 最大加速度 | 1-2 m/s² |
| 生成成功率 | 50-70% (当前) |

---

## 📚 文档

- [**详细使用指南**](docs/README.md) - 完整的使用文档
- [**项目总结**](docs/SUMMARY.md) - 功能总结和数据格式
- [**C++ 算法说明**](src/my_simple_planner/README.md) - 算法实现细节

---

## 🛠️ 开发

### 运行测试
```bash
python3 verify_trajectory_data.py
```

### 生成大规模数据集
```python
# 修改 standalone_trajectory_generator.py 中的参数
generator.generate_dataset(num_episodes=1000)
```

### 调试
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- 基于 Kinodynamic RRT* 算法
- 使用 Eigen3 线性代数库
- 兼容 gym-pybullet-drones

---

## 📞 联系方式

- **问题反馈**: 提交 GitHub Issue
- **功能建议**: 欢迎 Pull Request

---

**🎉 开始生成你的第一个轨迹吧！**

```bash
python3 standalone_trajectory_generator.py
```

#!/usr/bin/env python3
"""
虚拟环境 - 障碍物地图生成器
支持多种障碍物类型：盒子、球体、柱体等
"""

import json
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


class ObstacleType(Enum):
    """障碍物类型"""
    BOX = "box"              # 立方体/长方体
    SPHERE = "sphere"        # 球体
    CYLINDER = "cylinder"    # 圆柱体
    ELLIPSOID = "ellipsoid"  # 椭球体


class ObstacleMapGenerator:
    """
    障碍物地图生成器
    生成 3D 占据栅格地图
    """

    def __init__(
        self,
        map_size: Tuple[float, float, float] = (20.0, 20.0, 10.0),
        resolution: float = 0.2,
    ):
        """
        初始化地图生成器

        Args:
            map_size: 地图尺寸 (x, y, z) 单位：米
            resolution: 栅格分辨率，单位：米
        """
        self.map_size = np.array(map_size)
        self.resolution = resolution

        # 计算栅格尺寸
        self.grid_size = (self.map_size / resolution).astype(int)

        # 初始化占据栅格 (0=自由空间, 1=障碍物)
        self.obstacle_map = np.zeros(self.grid_size, dtype=np.uint8)

        # 记录所有添加的障碍物
        self.obstacles: List[Dict] = []

        # ESDF（欧式有符号距离场），单位：米；
        # 正值表示自由空间到最近障碍物的距离，负值表示在障碍物内部到自由边界的距离
        self.esdf = None

    def clear(self):
        """清空地图"""
        self.obstacle_map.fill(0)
        self.obstacles.clear()
        self.esdf = None

    def world_to_grid(self, position: np.ndarray) -> np.ndarray:
        """世界坐标转栅格坐标"""
        return (position / self.resolution).astype(int)

    def grid_to_world(self, grid_idx: np.ndarray) -> np.ndarray:
        """栅格坐标转世界坐标"""
        return grid_idx * self.resolution + self.resolution / 2.0

    def is_in_bounds(self, grid_idx: np.ndarray) -> bool:
        """检查栅格索引是否在地图范围内"""
        return np.all(grid_idx >= 0) and np.all(grid_idx < self.grid_size)

    # ==================== ESDF 构建与查询 ====================

    def build_esdf(self) -> np.ndarray:
        """构建 ESDF（欧式有符号距离场）

        返回：与 obstacle_map 同形状的 np.ndarray，dtype=float32，单位米
        约定：
          - 自由空间为正值：到最近障碍物表面的欧氏距离
          - 障碍物内部为负值：到最近自由空间边界的欧氏距离（取负号）
        """
        # free = True 表示自由体素；occupied = True 表示障碍体素
        free = (self.obstacle_map == 0)
        # 到最近“0”的距离，因此：
        # 1) 自由距离：把自由当作1，障碍当作0，则距离的是最近障碍
        dist_out = ndi.distance_transform_edt(free)
        # 2) 内部距离：把障碍当作1，自由当作0，则距离的是最近自由
        dist_in = ndi.distance_transform_edt(~free)

        esdf = dist_out.astype(np.float32) * float(self.resolution)
        esdf[~free] = -dist_in[~free].astype(np.float32) * float(self.resolution)
        self.esdf = esdf
        return self.esdf

    def world_to_grid_float(self, position: np.ndarray) -> np.ndarray:
        """世界坐标转为浮点网格坐标（未取整，体素中心在整数坐标+0.5附近）"""
        return position / self.resolution

    def get_distance_at_world(self, position: np.ndarray, clamp: bool = True) -> float:
        """查询给定世界坐标处的 ESDF 距离（米），采用三线性插值。

        注意：在 build_esdf() 被调用之前，esdf 为空，将自动构建。
        """
        if self.esdf is None:
            self.build_esdf()

        p = np.asarray(position, dtype=np.float64)
        g = self.world_to_grid_float(p) - 0.5  # 因为 grid_to_world 使用中心偏移 0.5
        x, y, z = g

        if clamp:
            x = np.clip(x, 0.0, self.grid_size[0] - 1.001)
            y = np.clip(y, 0.0, self.grid_size[1] - 1.001)
            z = np.clip(z, 0.0, self.grid_size[2] - 1.001)

        x0 = int(np.floor(x)); x1 = min(x0 + 1, self.grid_size[0] - 1)
        y0 = int(np.floor(y)); y1 = min(y0 + 1, self.grid_size[1] - 1)
        z0 = int(np.floor(z)); z1 = min(z0 + 1, self.grid_size[2] - 1)
        xd = float(x - x0); yd = float(y - y0); zd = float(z - z0)

        c000 = self.esdf[x0, y0, z0]; c100 = self.esdf[x1, y0, z0]
        c010 = self.esdf[x0, y1, z0]; c110 = self.esdf[x1, y1, z0]
        c001 = self.esdf[x0, y0, z1]; c101 = self.esdf[x1, y0, z1]
        c011 = self.esdf[x0, y1, z1]; c111 = self.esdf[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        c = c0 * (1 - zd) + c1 * zd
        return float(c)

    # ==================== 添加不同形状的障碍物 ====================

    def add_box_obstacle(
        self,
        center: np.ndarray,
        size: np.ndarray,
        label: str = "",
    ):
        """
        添加立方体/长方体障碍物

        Args:
            center: 中心位置 [x, y, z]
            size: 尺寸 [length_x, length_y, height_z]
            label: 障碍物标签
        """
        min_idx = np.maximum(0, self.world_to_grid(center - size / 2))
        max_idx = np.minimum(self.grid_size, self.world_to_grid(center + size / 2))

        self.obstacle_map[
            min_idx[0] : max_idx[0],
            min_idx[1] : max_idx[1],
            min_idx[2] : max_idx[2],
        ] = 1

        self.obstacles.append(
            {
                "type": ObstacleType.BOX.value,
                "center": center.tolist(),
                "size": size.tolist(),
                "label": label,
            }
        )

    def add_sphere_obstacle(
        self,
        center: np.ndarray,
        radius: float,
        label: str = "",
    ):
        """
        添加球体障碍物

        Args:
            center: 中心位置 [x, y, z]
            radius: 半径
            label: 障碍物标签
        """
        min_idx = np.maximum(0, self.world_to_grid(center - radius))
        max_idx = np.minimum(self.grid_size, self.world_to_grid(center + radius))

        for i in range(min_idx[0], max_idx[0]):
            for j in range(min_idx[1], max_idx[1]):
                for k in range(min_idx[2], max_idx[2]):
                    grid_pos = np.array([i, j, k])
                    world_pos = self.grid_to_world(grid_pos)

                    # 计算到球心的距离
                    dist = np.linalg.norm(world_pos - center)
                    if dist <= radius:
                        self.obstacle_map[i, j, k] = 1

        self.obstacles.append(
            {
                "type": ObstacleType.SPHERE.value,
                "center": center.tolist(),
                "radius": float(radius),
                "label": label,
            }
        )

    def add_cylinder_obstacle(
        self,
        center: np.ndarray,
        radius: float,
        height: float,
        label: str = "",
    ):
        """
        添加圆柱体障碍物（垂直于 z 轴）

        Args:
            center: 底面中心位置 [x, y, z]
            radius: 半径
            height: 高度
            label: 障碍物标签
        """
        min_idx = np.maximum(
            0, self.world_to_grid(center - np.array([radius, radius, 0]))
        )
        max_idx = np.minimum(
            self.grid_size, self.world_to_grid(center + np.array([radius, radius, height]))
        )

        for i in range(min_idx[0], max_idx[0]):
            for j in range(min_idx[1], max_idx[1]):
                for k in range(min_idx[2], max_idx[2]):
                    grid_pos = np.array([i, j, k])
                    world_pos = self.grid_to_world(grid_pos)

                    # 检查是否在圆柱体内（xy平面距离 + z高度）
                    xy_dist = np.linalg.norm(world_pos[:2] - center[:2])
                    if xy_dist <= radius and center[2] <= world_pos[2] <= center[2] + height:
                        self.obstacle_map[i, j, k] = 1

        self.obstacles.append(
            {
                "type": ObstacleType.CYLINDER.value,
                "center": center.tolist(),
                "radius": float(radius),
                "height": float(height),
                "label": label,
            }
        )

    def add_ellipsoid_obstacle(
        self,
        center: np.ndarray,
        semi_axes: np.ndarray,
        label: str = "",
    ):
        """
        添加椭球体障碍物

        Args:
            center: 中心位置 [x, y, z]
            semi_axes: 三个半轴长度 [a, b, c]
            label: 障碍物标签
        """
        min_idx = np.maximum(0, self.world_to_grid(center - semi_axes))
        max_idx = np.minimum(self.grid_size, self.world_to_grid(center + semi_axes))

        for i in range(min_idx[0], max_idx[0]):
            for j in range(min_idx[1], max_idx[1]):
                for k in range(min_idx[2], max_idx[2]):
                    grid_pos = np.array([i, j, k])
                    world_pos = self.grid_to_world(grid_pos)

                    # 椭球体方程: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
                    delta = world_pos - center
                    normalized = delta / semi_axes
                    if np.sum(normalized**2) <= 1.0:
                        self.obstacle_map[i, j, k] = 1

        self.obstacles.append(
            {
                "type": ObstacleType.ELLIPSOID.value,
                "center": center.tolist(),
                "semi_axes": semi_axes.tolist(),
                "label": label,
            }
        )

    # ==================== 批量生成障碍物 ====================

    def generate_random_boxes(
        self,
        num_obstacles: int = 20,
        size_range: Tuple[float, float] = (1.0, 4.0),
        safe_margin: float = 2.0,
    ):
        """
        生成随机立方体障碍物

        Args:
            num_obstacles: 障碍物数量
            size_range: 尺寸范围 (最小, 最大)
            safe_margin: 边界安全距离
        """
        for i in range(num_obstacles):
            # 随机中心位置（避开边界）
            center = np.random.rand(3) * (self.map_size - 2 * safe_margin) + safe_margin

            # 随机尺寸
            size = (
                np.random.rand(3) * (size_range[1] - size_range[0]) + size_range[0]
            )

            self.add_box_obstacle(center, size, label=f"box_{i}")

    def generate_random_spheres(
        self,
        num_obstacles: int = 15,
        radius_range: Tuple[float, float] = (0.5, 2.0),
        safe_margin: float = 2.0,
    ):
        """
        生成随机球体障碍物

        Args:
            num_obstacles: 障碍物数量
            radius_range: 半径范围 (最小, 最大)
            safe_margin: 边界安全距离
        """
        for i in range(num_obstacles):
            center = np.random.rand(3) * (self.map_size - 2 * safe_margin) + safe_margin

            radius = (
                np.random.rand() * (radius_range[1] - radius_range[0]) + radius_range[0]
            )

            self.add_sphere_obstacle(center, radius, label=f"sphere_{i}")

    def generate_random_cylinders(
        self,
        num_obstacles: int = 10,
        radius_range: Tuple[float, float] = (0.5, 1.5),
        height_range: Tuple[float, float] = (2.0, 6.0),
        safe_margin: float = 2.0,
    ):
        """
        生成随机圆柱体障碍物

        Args:
            num_obstacles: 障碍物数量
            radius_range: 半径范围
            height_range: 高度范围
            safe_margin: 边界安全距离
        """
        for i in range(num_obstacles):
            # 圆柱体底面中心
            center = np.random.rand(3) * (self.map_size - 2 * safe_margin) + safe_margin
            center[2] = 0  # 底面在地面

            radius = (
                np.random.rand() * (radius_range[1] - radius_range[0]) + radius_range[0]
            )
            height = (
                np.random.rand() * (height_range[1] - height_range[0]) + height_range[0]
            )

            self.add_cylinder_obstacle(center, radius, height, label=f"cylinder_{i}")

    def generate_forest_environment(
        self,
        num_trees: int = 30,
        trunk_radius_range: Tuple[float, float] = (0.3, 0.8),
        tree_height_range: Tuple[float, float] = (3.0, 8.0),
    ):
        """
        生成森林环境（树木 = 圆柱体）

        Args:
            num_trees: 树木数量
            trunk_radius_range: 树干半径范围
            tree_height_range: 树高范围
        """
        print(f"生成森林环境：{num_trees} 棵树")
        self.generate_random_cylinders(
            num_obstacles=num_trees,
            radius_range=trunk_radius_range,
            height_range=tree_height_range,
            safe_margin=1.5,
        )

    def generate_urban_environment(
        self,
        num_buildings: int = 15,
        building_size_range: Tuple[float, float] = (3.0, 8.0),
    ):
        """
        生成城市环境（建筑物 = 长方体）

        Args:
            num_buildings: 建筑物数量
            building_size_range: 建筑物尺寸范围
        """
        print(f"生成城市环境：{num_buildings} 栋建筑")
        self.generate_random_boxes(
            num_obstacles=num_buildings,
            size_range=building_size_range,
            safe_margin=2.0,
        )

    def generate_building_city_environment(
        self,
        num_buildings: int = 20,
        footprint_range: Tuple[float, float] = (2.0, 6.0),
        height_range: Tuple[float, float] = (3.0, 10.0),
        min_gap: float = 1.0,
        boundary_margin: float = 1.0,
        max_tries_per_building: int = 200,
        random_seed: Optional[int] = None,
    ):
        """生成“楼宇”环境（仅长方体建筑，底面落在地面 z=0）。

        与 generate_urban_environment 的区别：
        - 本函数把建筑当作“楼宇”：底面固定在地面，尺寸明确为 (footprint_x, footprint_y, height_z)
        - 在 XY 平面做简单的 AABB 非重叠约束，确保建筑之间留有最小间距（可理解为街道/通行空间）

        Args:
            num_buildings: 楼宇数量
            footprint_range: 楼宇底面边长范围（米），x/y 两个方向各自独立采样
            height_range: 楼宇高度范围（米）
            min_gap: 楼宇之间最小间距（米），在 XY 平面生效
            boundary_margin: 与地图边界的最小距离（米），避免贴边
            max_tries_per_building: 单栋楼宇的采样重试次数上限
            random_seed: 随机种子（可选）
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        print(f"生成楼宇环境：{num_buildings} 栋楼宇（仅长方体）")

        placed_xy: List[Tuple[float, float, float, float]] = []  # (xmin, xmax, ymin, ymax)

        def overlaps(a: Tuple[float, float, float, float]) -> bool:
            ax0, ax1, ay0, ay1 = a
            for bx0, bx1, by0, by1 in placed_xy:
                # AABB overlap in XY
                if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
                    return True
            return False

        for i in range(num_buildings):
            placed = False
            for _ in range(max_tries_per_building):
                sx = float(np.random.uniform(footprint_range[0], footprint_range[1]))
                sy = float(np.random.uniform(footprint_range[0], footprint_range[1]))
                h = float(np.random.uniform(height_range[0], height_range[1]))

                x_min = boundary_margin + sx / 2
                x_max = float(self.map_size[0]) - boundary_margin - sx / 2
                y_min = boundary_margin + sy / 2
                y_max = float(self.map_size[1]) - boundary_margin - sy / 2
                if x_max <= x_min or y_max <= y_min:
                    break

                cx = float(np.random.uniform(x_min, x_max))
                cy = float(np.random.uniform(y_min, y_max))

                # 在 XY 平面膨胀一个 min_gap/2 作为“安全间距”
                pad = min_gap / 2.0
                rect = (cx - sx / 2 - pad, cx + sx / 2 + pad, cy - sy / 2 - pad, cy + sy / 2 + pad)
                if overlaps(rect):
                    continue

                center = np.array([cx, cy, h / 2.0], dtype=float)
                size = np.array([sx, sy, h], dtype=float)
                self.add_box_obstacle(center=center, size=size, label=f"building_{i}")
                placed_xy.append(rect)
                placed = True
                break

            if not placed:
                # 放不下就提前结束，避免死循环
                print(f"警告: 第 {i} 栋楼宇放置失败（地图太挤或参数不合适），已提前停止。")
                break

    # ==================== 检查与查询 ====================

    def is_collision_free(self, position: np.ndarray) -> bool:
        """检查某点是否无碰撞"""
        idx = self.world_to_grid(position)

        if not self.is_in_bounds(idx):
            return False

        return self.obstacle_map[idx[0], idx[1], idx[2]] == 0

    def check_path_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_checks: int = 50,
    ) -> bool:
        """检查路径是否无碰撞"""
        for t in np.linspace(0, 1, num_checks):
            pos = start + t * (end - start)
            if not self.is_collision_free(pos):
                return False
        return True

    def get_occupancy_count(self) -> int:
        """获取占用体素数量"""
        return int(np.sum(self.obstacle_map))

    def get_occupancy_ratio(self) -> float:
        """获取占用率"""
        total = int(np.prod(self.grid_size))
        occupied = self.get_occupancy_count()
        return occupied / total if total > 0 else 0.0

    # ==================== 保存与加载 ====================

    def save(self, filepath: str):
        """保存地图到文件"""
        data = {
            "map_size": self.map_size.tolist(),
            "resolution": self.resolution,
            "grid_size": self.grid_size.tolist(),
            "obstacle_map": self.obstacle_map,
            "obstacles": self.obstacles,
        }
        if self.esdf is not None:
            data["esdf"] = self.esdf

        if filepath.endswith(".npz"):
            np.savez_compressed(filepath, **data)
            print(f"地图已保存到: {filepath}")
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ObstacleMapGenerator":
        """从文件加载地图"""
        if filepath.endswith(".npz"):
            data = np.load(filepath, allow_pickle=True)

            generator = cls(
                map_size=tuple(data["map_size"]),
                resolution=float(data["resolution"]),
            )
            generator.obstacle_map = data["obstacle_map"]
            generator.obstacles = list(data["obstacles"])
            if "esdf" in data:
                generator.esdf = data["esdf"].astype(np.float32)

            print(f"地图已加载: {filepath}")
            return generator
        raise ValueError(f"不支持的文件格式: {filepath}")

    def get_summary(self) -> Dict:
        """获取地图摘要信息"""
        return {
            "map_size": self.map_size.tolist(),
            "resolution": self.resolution,
            "grid_size": self.grid_size.tolist(),
            "num_obstacles": len(self.obstacles),
            "occupied_voxels": self.get_occupancy_count(),
            "occupancy_ratio": f"{self.get_occupancy_ratio() * 100:.2f}%",
            "has_esdf": self.esdf is not None,
            "obstacle_types": {
                obs_type: sum(1 for o in self.obstacles if o["type"] == obs_type)
                for obs_type in {o["type"] for o in self.obstacles}
            },
        }

    def print_summary(self):
        """打印地图摘要"""
        summary = self.get_summary()
        print("\n========== 地图摘要 ==========")
        print(f"地图尺寸: {summary['map_size']} m")
        print(f"分辨率: {summary['resolution']} m")
        print(f"栅格尺寸: {summary['grid_size']}")
        print(f"障碍物数量: {summary['num_obstacles']}")
        print(f"占用体素数: {summary['occupied_voxels']}")
        print(f"占用率: {summary['occupancy_ratio']}")
        print(f"障碍物类型分布: {summary['obstacle_types']}")
        print("=============================\n")

    # ==================== 导出：占据栅格 ASCII + ESDF NPZ ====================

    def export_ascii_occupancy(self, filepath: str, chunk: int = 64):
        """将占据栅格导出为 ASCII 文本，方便 C++ 可执行程序直接读取。

        文件格式：
          1) nx ny nz
          2) resolution
          3) origin_x origin_y origin_z （此处采用 0 0 0，表示以地图中心为世界原点）
          4+) nx*ny*nz 个 0/1 值（空白分隔），顺序为 C-order 扁平化
        """
        nx, ny, nz = map(int, self.grid_size)
        with open(filepath, 'w') as f:
            f.write(f"{nx} {ny} {nz}\n")
            f.write(f"{float(self.resolution)}\n")
            f.write("0 0 0\n")
            flat = self.obstacle_map.astype(np.int32).ravel(order='C')
            for i in range(0, flat.size, chunk):
                line = " ".join(str(int(v)) for v in flat[i:i+chunk])
                f.write(line + "\n")

    def export_dual_maps(self, basepath: str):
        """导出两套地图文件：
        - 占据栅格 ASCII：{basepath}_occ.txt （快速碰撞检测）
        - ESDF 打包 NPZ：{basepath}_esdf.npz （轨迹优化/距离代价）
        """
        # 1) 占据栅格 ASCII
        occ_path = f"{basepath}_occ.txt"
        self.export_ascii_occupancy(occ_path)

        # 2) ESDF NPZ（若尚未构建 ESDF，则先构建）
        if self.esdf is None:
            self.build_esdf()
        esdf_path = f"{basepath}_esdf.npz"
        data = {
            "map_size": self.map_size.astype(float),
            "resolution": float(self.resolution),
            "grid_size": self.grid_size.astype(int),
            "obstacle_map": self.obstacle_map.astype(np.uint8),
            "esdf": self.esdf.astype(np.float32),
        }
        np.savez_compressed(esdf_path, **data)

    def export_esdf_ascii(self, filepath: str, chunk: int = 32):
        """将 ESDF 导出为 ASCII 文本，便于 C++ 侧（无需第三方库）直接读取。

        文件格式：
          1) nx ny nz
          2) resolution
          3) origin_x origin_y origin_z （此处采用 0 0 0，表示以地图中心为世界原点）
          4+) nx*ny*nz 个浮点值（米），空白分隔，C-order 扁平化
        """
        if self.esdf is None:
            self.build_esdf()
        nx, ny, nz = map(int, self.grid_size)
        with open(filepath, 'w') as f:
            f.write(f"{nx} {ny} {nz}\n")
            f.write(f"{float(self.resolution)}\n")
            f.write("0 0 0\n")
            flat = self.esdf.astype(np.float32).ravel(order='C')
            for i in range(0, flat.size, chunk):
                line = " ".join(f"{float(v):.6f}" for v in flat[i:i+chunk])
                f.write(line + "\n")

    def export_dual_npy(self, basepath: str):
        """导出两套 NPY 文件（仅数组）：
        - {basepath}_occ.npy  (uint8，占据：0/1)
        - {basepath}_esdf.npy (float32，单位米)

        说明：
        - 分辨率（cell size）与原点约定：
          * 原点使用 (0,0,0) 在地图中心（与 ASCII 导出一致）；
          * 分辨率请在 C++ 可执行程序中通过 --grid-res 传入；
        - 这样做可以保持 C++ 端零依赖、快速加载。
        """
        occ_path = f"{basepath}_occ.npy"
        esdf_path = f"{basepath}_esdf.npy"
        if self.esdf is None:
            self.build_esdf()
        np.save(occ_path, self.obstacle_map.astype(np.uint8), allow_pickle=False)
        np.save(esdf_path, self.esdf.astype(np.float32), allow_pickle=False)


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    print("========== 障碍物地图生成器示例 ==========\n")

    # 创建地图生成器
    generator = ObstacleMapGenerator(
        map_size=(20.0, 20.0, 10.0),
        resolution=0.2,
    )

    # 示例1：手动添加障碍物
    print("1. 手动添加障碍物")
    generator.add_box_obstacle(
        center=np.array([5.0, 5.0, 2.5]),
        size=np.array([2.0, 2.0, 5.0]),
        label="building_1",
    )

    generator.add_sphere_obstacle(
        center=np.array([10.0, 8.0, 3.0]),
        radius=1.5,
        label="sphere_1",
    )

    generator.add_cylinder_obstacle(
        center=np.array([15.0, 5.0, 0.0]),
        radius=0.8,
        height=6.0,
        label="tree_1",
    )

    # 示例2：批量生成随机障碍物
    print("\n2. 批量生成随机障碍物")
    generator.generate_random_boxes(num_obstacles=5)
    generator.generate_random_spheres(num_obstacles=3)

    # 打印摘要
    generator.print_summary()

    # 示例3：碰撞检测
    print("3. 碰撞检测")
    test_point = np.array([5.0, 5.0, 2.5])
    is_free = generator.is_collision_free(test_point)
    print(f"点 {test_point} 是否无碰撞: {is_free}")

    # 示例4：保存地图
    print("\n4. 保存地图")
    generator.save("example_map.npz")

    # 示例5：加载地图
    print("\n5. 加载地图")
    loaded_generator = ObstacleMapGenerator.load("example_map.npz")
    loaded_generator.print_summary()

    return generator


if __name__ == "__main__":
    example_usage()

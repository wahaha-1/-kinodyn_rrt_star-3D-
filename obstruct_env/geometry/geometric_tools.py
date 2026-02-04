#!/usr/bin/env python3
"""几何实体工具"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from obstruct_env.core.obstacle_map import ObstacleMapGenerator


class GeometricObstacle:
    def __init__(self, obstacle_type: str, center: np.ndarray, label: str = ""):
        self.type = obstacle_type
        self.center = np.array(center)
        self.label = label

    def contains_point(self, point: np.ndarray) -> bool:
        raise NotImplementedError

    def distance_to_point(self, point: np.ndarray) -> float:
        raise NotImplementedError

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError


class BoxObstacle(GeometricObstacle):
    def __init__(self, center: np.ndarray, size: np.ndarray, label: str = ""):
        super().__init__("box", center, label)
        self.size = np.array(size)
        self.half_size = self.size / 2.0

    def contains_point(self, point: np.ndarray) -> bool:
        diff = np.abs(point - self.center)
        return np.all(diff <= self.half_size)

    def distance_to_point(self, point: np.ndarray) -> float:
        diff = np.abs(point - self.center) - self.half_size
        if np.all(diff <= 0):
            return np.max(diff)
        outside = np.maximum(diff, 0)
        return np.linalg.norm(outside)

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.center - self.half_size, self.center + self.half_size

    def to_dict(self) -> dict:
        return {
            "type": "box",
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "label": self.label,
        }


class SphereObstacle(GeometricObstacle):
    def __init__(self, center: np.ndarray, radius: float, label: str = ""):
        super().__init__("sphere", center, label)
        self.radius = radius

    def contains_point(self, point: np.ndarray) -> bool:
        return np.linalg.norm(point - self.center) <= self.radius

    def distance_to_point(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.center) - self.radius

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.center - self.radius, self.center + self.radius

    def to_dict(self) -> dict:
        return {
            "type": "sphere",
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "label": self.label,
        }


class CylinderObstacle(GeometricObstacle):
    def __init__(self, center: np.ndarray, radius: float, height: float, label: str = ""):
        super().__init__("cylinder", center, label)
        self.radius = radius
        self.height = height
        self.half_height = height / 2.0

    def contains_point(self, point: np.ndarray) -> bool:
        if abs(point[2] - self.center[2]) > self.half_height:
            return False
        return np.linalg.norm(point[:2] - self.center[:2]) <= self.radius

    def distance_to_point(self, point: np.ndarray) -> float:
        z_dist = abs(point[2] - self.center[2]) - self.half_height
        xy_dist = np.linalg.norm(point[:2] - self.center[:2]) - self.radius
        if z_dist <= 0 and xy_dist <= 0:
            return max(z_dist, xy_dist)
        return np.sqrt(max(z_dist, 0) ** 2 + max(xy_dist, 0) ** 2)

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.center - np.array([self.radius, self.radius, self.half_height]),
            self.center + np.array([self.radius, self.radius, self.half_height]),
        )

    def to_dict(self) -> dict:
        return {
            "type": "cylinder",
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "height": float(self.height),
            "label": self.label,
        }


class EllipsoidObstacle(GeometricObstacle):
    def __init__(self, center: np.ndarray, radii: np.ndarray, label: str = ""):
        super().__init__("ellipsoid", center, label)
        self.radii = np.array(radii)

    def contains_point(self, point: np.ndarray) -> bool:
        diff = (point - self.center) / self.radii
        return np.linalg.norm(diff) <= 1.0

    def distance_to_point(self, point: np.ndarray) -> float:
        diff = (point - self.center) / self.radii
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return -np.min(self.radii)
        direction = diff / norm
        surface_point = self.center + direction * self.radii
        dist = np.linalg.norm(point - surface_point)
        return dist if norm > 1.0 else -dist

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.center - self.radii, self.center + self.radii

    def to_dict(self) -> dict:
        return {
            "type": "ellipsoid",
            "center": self.center.tolist(),
            "radii": self.radii.tolist(),
            "label": self.label,
        }


class GeometricEnvironment:
    def __init__(self, map_size: Tuple[float, float, float] = (20.0, 20.0, 10.0)):
        self.map_size = np.array(map_size)
        self.obstacles: List[GeometricObstacle] = []

    @classmethod
    def from_obstacle_map_generator(cls, generator: ObstacleMapGenerator) -> "GeometricEnvironment":
        env = cls(map_size=tuple(generator.map_size))
        for obs_dict in generator.obstacles:
            obs = cls._create_obstacle_from_dict(obs_dict)
            if obs:
                env.obstacles.append(obs)
        return env

    @staticmethod
    def _create_obstacle_from_dict(obs_dict: Dict) -> Optional[GeometricObstacle]:
        obs_type = obs_dict["type"]
        center = np.array(obs_dict["center"])
        label = obs_dict.get("label", "")
        if obs_type == "box":
            return BoxObstacle(center, np.array(obs_dict["size"]), label)
        if obs_type == "sphere":
            return SphereObstacle(center, obs_dict["radius"], label)
        if obs_type == "cylinder":
            return CylinderObstacle(center, obs_dict["radius"], obs_dict["height"], label)
        if obs_type == "ellipsoid":
            return EllipsoidObstacle(center, np.array(obs_dict["semi_axes"]), label)
        print(f"未知障碍物类型: {obs_type}")
        return None

    def add_obstacle(self, obstacle: GeometricObstacle):
        self.obstacles.append(obstacle)

    def is_collision_free(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        for obstacle in self.obstacles:
            if obstacle.distance_to_point(point) < safety_margin:
                return False
        return True

    def get_nearest_obstacle_distance(self, point: np.ndarray) -> float:
        if not self.obstacles:
            return float("inf")
        return min(obstacle.distance_to_point(point) for obstacle in self.obstacles)

    def check_path_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_checks: int = 50,
        safety_margin: float = 0.0,
    ) -> bool:
        for t in np.linspace(0, 1, num_checks):
            point = start + t * (end - start)
            if not self.is_collision_free(point, safety_margin):
                return False
        return True

    def get_colliding_obstacles(self, point: np.ndarray) -> List[GeometricObstacle]:
        return [obs for obs in self.obstacles if obs.contains_point(point)]

    def export_to_json(self, filepath: str):
        data = {
            "map_size": self.map_size.tolist(),
            "num_obstacles": len(self.obstacles),
            "obstacles": [obs.to_dict() for obs in self.obstacles],
        }
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        print(f"✓ 已导出到: {filepath}")

    def export_to_sdf(self, filepath: str):
        content = [
            "<?xml version=\"1.0\"?>",
            "<sdf version=\"1.6\">",
            "  <world name=\"obstacle_world\">",
            "    <scene>",
            "      <ambient>0.4 0.4 0.4 1</ambient>",
            "      <background>0.7 0.7 0.7 1</background>",
            "    </scene>",
            "    <include><uri>model://ground_plane</uri></include>",
            "    <include><uri>model://sun</uri></include>",
            "",
        ]
        for index, obs in enumerate(self.obstacles):
            if obs.type == "box":
                content.append(self._box_to_sdf(obs, index))
            elif obs.type == "sphere":
                content.append(self._sphere_to_sdf(obs, index))
            elif obs.type == "cylinder":
                content.append(self._cylinder_to_sdf(obs, index))
        content.append("  </world>")
        content.append("</sdf>")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as sdf_file:
            sdf_file.write("\n".join(content))
        print(f"✓ 已导出为SDF: {filepath}")

    def _box_to_sdf(self, obs: BoxObstacle, index: int) -> str:
        x, y, z = obs.center
        sx, sy, sz = obs.size
        return f"    <model name=\"box_{index}\">\n      <static>true</static>\n      <pose>{x} {y} {z} 0 0 0</pose>\n      <link name=\"link\">\n        <collision name=\"collision\">\n          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>\n        </collision>\n        <visual name=\"visual\">\n          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>\n          <material><ambient>0.8 0.2 0.2 1</ambient></material>\n        </visual>\n      </link>\n    </model>"

    def _sphere_to_sdf(self, obs: SphereObstacle, index: int) -> str:
        x, y, z = obs.center
        r = obs.radius
        return f"    <model name=\"sphere_{index}\">\n      <static>true</static>\n      <pose>{x} {y} {z} 0 0 0</pose>\n      <link name=\"link\">\n        <collision name=\"collision\">\n          <geometry><sphere><radius>{r}</radius></sphere></geometry>\n        </collision>\n        <visual name=\"visual\">\n          <geometry><sphere><radius>{r}</radius></sphere></geometry>\n          <material><ambient>0.2 0.8 0.2 1</ambient></material>\n        </visual>\n      </link>\n    </model>"

    def _cylinder_to_sdf(self, obs: CylinderObstacle, index: int) -> str:
        x, y, z = obs.center
        r = obs.radius
        h = obs.height
        return f"    <model name=\"cylinder_{index}\">\n      <static>true</static>\n      <pose>{x} {y} {z} 0 0 0</pose>\n      <link name=\"link\">\n        <collision name=\"collision\">\n          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>\n        </collision>\n        <visual name=\"visual\">\n          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>\n          <material><ambient>0.2 0.2 0.8 1</ambient></material>\n        </visual>\n      </link>\n    </model>"

    def get_statistics(self) -> dict:
        type_counts: Dict[str, int] = {}
        for obs in self.obstacles:
            type_counts[obs.type] = type_counts.get(obs.type, 0) + 1
        return {
            "map_size": self.map_size.tolist(),
            "num_obstacles": len(self.obstacles),
            "obstacle_types": type_counts,
        }

    def print_summary(self):
        stats = self.get_statistics()
        print("\n========== 几何环境摘要 ==========")
        print(f"地图尺寸: {stats['map_size']} m")
        print(f"障碍物总数: {stats['num_obstacles']}")
        print("障碍物类型分布:")
        for obs_type, count in stats["obstacle_types"].items():
            print(f"  {obs_type}: {count}")


def example_usage():
    print("========== 几何实体工具示例 ==========\n")
    map_gen = ObstacleMapGenerator(map_size=(20, 20, 10), resolution=0.2)
    map_gen.add_box_obstacle(np.array([10, 10, 5]), np.array([3, 3, 4]))
    map_gen.add_sphere_obstacle(np.array([5, 5, 3]), 1.5)
    map_gen.add_cylinder_obstacle(np.array([15, 5, 0.0]), 1.0, 5.0)

    geo_env = GeometricEnvironment.from_obstacle_map_generator(map_gen)
    geo_env.print_summary()

    test_point = np.array([10.0, 10.0, 5.0])
    is_free = geo_env.is_collision_free(test_point, safety_margin=0.5)
    nearest_dist = geo_env.get_nearest_obstacle_distance(test_point)
    print(f"测试点 {test_point}: {'无碰撞' if is_free else '有碰撞'}")
    print(f"到最近障碍物距离: {nearest_dist:.3f}m")

    start = np.array([1.0, 1.0, 5.0])
    goal = np.array([19.0, 19.0, 5.0])
    is_path_free = geo_env.check_path_collision(start, goal, num_checks=100)
    print(f"路径 {start} -> {goal}: {'无碰撞' if is_path_free else '有碰撞'}")

    output_dir = Path("./obstruct_env/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    geo_env.export_to_json(str(output_dir / "geometric_env.json"))
    geo_env.export_to_sdf(str(output_dir / "gazebo_world.sdf"))

    print("\n示例完成!")


if __name__ == "__main__":
    example_usage()

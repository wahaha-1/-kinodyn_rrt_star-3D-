#!/usr/bin/env python3
"""
ESDF 生成器：基于占据栅格 obstacle_map 计算欧式有符号距离场（ESDF）。
- 自由体素为正值：到最近障碍表面的欧氏距离（米）
- 障碍体素为负值：到最近自由边界的欧氏距离（米，取负）

依赖：scipy.ndimage.distance_transform_edt
"""
from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
from scipy import ndimage as ndi


class EnvironmentGenerator:
    """环境生成相关工具：ESDF 计算与保存。"""

    def compute_distance_transform(self, obstacle_map: np.ndarray, resolution: float) -> np.ndarray:
        """计算有符号距离场（ESDF）。

        Args:
            obstacle_map: 3D uint8 数组，1=障碍，0=自由
            resolution: 体素边长（米）
        Returns:
            esdf: 3D float32 数组（米），正/负号如上所述
        """
        if obstacle_map.ndim != 3:
            raise ValueError("obstacle_map 必须是 3 维数组 (nx, ny, nz)")

        free = obstacle_map == 0
        # 自由到障碍的距离
        dist_out = ndi.distance_transform_edt(free)
        # 障碍到自由的距离
        dist_in = ndi.distance_transform_edt(~free)

        esdf = dist_out.astype(np.float32) * float(resolution)
        esdf[~free] = -dist_in[~free].astype(np.float32) * float(resolution)
        return esdf

    def generate_esdf(self, obstacle_map: np.ndarray, resolution: float) -> np.ndarray:
        """对外入口：基于 obstacle_map 生成 ESDF。"""
        return self.compute_distance_transform(obstacle_map, resolution)

    def save_npz(
        self,
        filepath: str,
        obstacle_map: np.ndarray,
        resolution: float,
        esdf: Optional[np.ndarray] = None,
        map_size: Optional[np.ndarray] = None,
        grid_size: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存 obstacle_map 与 esdf 到 .npz，便于后续规划/优化。

        存储键：
          - obstacle_map (uint8)
          - resolution (float)
          - esdf (float32) 如果提供
          - map_size (3,) 如果提供（米）
          - grid_size (3,) 如果提供（体素数）
          - 以及 extra 字典中传入的自定义键
        """
        data: Dict[str, Any] = {
            "obstacle_map": obstacle_map,
            "resolution": float(resolution),
        }
        if esdf is not None:
            data["esdf"] = esdf
        if map_size is not None:
            data["map_size"] = np.asarray(map_size, dtype=np.float32)
        if grid_size is not None:
            data["grid_size"] = np.asarray(grid_size, dtype=np.int32)
        if extra:
            for k, v in extra.items():
                data[k] = v
        np.savez_compressed(filepath, **data)

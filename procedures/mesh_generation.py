import numpy as np
import utils.utils as utils
import utils.mesh as mesh
from numpy.typing import NDArray
from utils.utils import TriangleArray, PointArray


def gen_mesh_from_pts(points: NDArray) -> tuple[NDArray, NDArray]:
    """
    从给定点生成三角形网格

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)

    Returns:
        tuple[NDArray, NDArray]: 三角形的顶点坐标和三角形的索引
    """
    pass


def refine_mesh(
    points: NDArray, triangles: NDArray, max_edge_length: float
) -> tuple[NDArray, NDArray]:
    """
    对三角形网格进行细化
    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)
        triangles (NDArray): 三角形的索引，形状为 (m, 3)
        max_edge_length (float): 最大边长

    Returns:
        tuple[NDArray, NDArray]: 细化后的三角形的顶点坐标和三角形的索引
    """
    pass

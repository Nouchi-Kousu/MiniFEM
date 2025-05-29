import numpy as np
from numpy.typing import NDArray
from scipy.linalg import det


def pt_in_circle(point: NDArray, triangle: NDArray, points: NDArray) -> NDArray:
    """
    检查点是否在三角形外接圆内（基于几何方法）

    Args:
        point (NDArray): 待检查的点，形状为 (2,)
        triangle (NDArray): 三角形的索引，形状为 (m, 3)
        points (NDArray): 所有点的坐标，形状为 (n, 2)

    Returns:
        NDArray: 布尔数组，表示点是否在每个三角形的外接圆内
    """
    assert triangle.shape[1] == 3, "三角形必须由三个点组成"
    assert point.shape == (2,), "检查点必须是二维坐标"
    assert points.shape[1] == 2, "点必须是二维坐标"
    assert np.all(triangle >= 0) and np.all(triangle < points.shape[0]), (
        "三角形索引超出点的范围"
    )

    # 获取三角形的顶点坐标
    a = points[triangle[:, 0]]
    b = points[triangle[:, 1]]
    c = points[triangle[:, 2]]

    # D = 2 * (a_x * (b_y - c_y) + b_x * (c_y - a_y) + c_x * (a_y - b_y))
    D = 2 * (
        a[:, 0] * (b[:, 1] - c[:, 1])
        + b[:, 0] * (c[:, 1] - a[:, 1])
        + c[:, 0] * (a[:, 1] - b[:, 1])
    )

    # 初始化结果数组
    is_in_circle = np.zeros(triangle.shape[0], dtype=bool)
    # 避免除以零，对于共线的点 (D=0)，它们没有定义的外接圆，或者半径无限大
    # 在这种情况下，我们认为点不在圆内
    non_collinear_mask = np.abs(D) > 1e-12

    # 只对非共线的三角形进行计算
    if np.any(non_collinear_mask):
        a_sq = np.sum(a[non_collinear_mask] ** 2, axis=1)
        b_sq = np.sum(b[non_collinear_mask] ** 2, axis=1)
        c_sq = np.sum(c[non_collinear_mask] ** 2, axis=1)

        current_a = a[non_collinear_mask]
        current_b = b[non_collinear_mask]
        current_c = c[non_collinear_mask]
        current_D = D[non_collinear_mask]

        # 外接圆圆心坐标 (Ux, Uy)
        Ux = (
            a_sq * (current_b[:, 1] - current_c[:, 1])
            + b_sq * (current_c[:, 1] - current_a[:, 1])
            + c_sq * (current_a[:, 1] - current_b[:, 1])
        ) / current_D
        Uy = (
            a_sq * (current_c[:, 0] - current_b[:, 0])
            + b_sq * (current_a[:, 0] - current_c[:, 0])
            + c_sq * (current_b[:, 0] - current_a[:, 0])
        ) / current_D

        circumcenters = np.stack((Ux, Uy), axis=1)

        # 外接圆半径的平方 R^2 = (a_x - Ux)^2 + (a_y - Uy)^2
        circumradius_sq = np.sum((current_a - circumcenters) ** 2, axis=1)

        # 点到圆心的距离的平方
        point_dist_sq = np.sum(
            (circumcenters - point) ** 2, axis=1
        )  # 'point' 是待检查的点

        is_in_circle[non_collinear_mask] = point_dist_sq < circumradius_sq

    return is_in_circle


def tris_unique_edges(triangles: NDArray) -> NDArray:
    """
    获取三角形的的唯一边集合, 删除有重复的边, 构建星形多边形

    Bowyer-Watson算法

    Args:
        triangles (NDArray): 定义三角形的三个点，形状为 (n, 3)

    Returns:
        NDArray: 三角形的唯一边集合，形状为 (m, 2)
    """

    assert triangles.shape[1] == 3, "三角形必须由三个点组成"

    # 获取三角形的边
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [0, 2]],
            triangles[:, [1, 2]],
        ],
        axis=0,
    )

    # 获取唯一边
    unique_edges, counts = np.unique(edges, return_counts=True, axis=0)

    return unique_edges[counts == 1]


def get_adj(triangles: NDArray) -> NDArray:
    """
    将三角形关系转变为单相图边列表

    Args:
        triangles (NDArray): 定义三角形的三个点，形状为 (n, 3)

    Returns:
        NDArray: 三角形的边集合，形状为 (m, 2)
    """
    assert triangles.shape[1] == 3, "三角形必须由三个点组成"

    # 获取三角形的边
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [0, 2]],
            triangles[:, [1, 2]],
        ],
        axis=0,
    )

    # 获取边集合
    unique_edges = np.unique(edges, axis=0)

    return unique_edges

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import det


def pt_in_circle(point: NDArray, triangle: NDArray) -> bool:
    """
    检查点是否在三角形外接圆内

    https://zhuanlan.zhihu.com/p/534117002

    Args:
        point (NDArray): 检查点
        triangle (NDArray): 定义三角形的三个点

    Returns:
        bool: 是否在三角形外接圆内
    """
    assert triangle.shape == (3, 2), "三角形必须由三个点组成"
    assert point.shape == (2,), "检查点必须是二维坐标"

    a_x, a_y = triangle[0][0], triangle[0][1]
    b_x, b_y = triangle[1][0], triangle[1][1]
    c_x, c_y = triangle[2][0], triangle[2][1]

    p_x, p_y = point[0], point[1]

    in_circle = det(
        np.array(
            [
                [a_x, a_y, a_x**2 + a_y**2, 1],
                [b_x, b_y, b_x**2 + b_y**2, 1],
                [c_x, c_y, c_x**2 + c_y**2, 1],
                [p_x, p_y, p_x**2 + p_y**2, 1],
            ]
        )
    )

    return float(in_circle) >= 0


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

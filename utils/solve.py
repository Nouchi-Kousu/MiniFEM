import numpy as np
from numpy.typing import NDArray
from utils.utils import tri_area


def a_list(points: NDArray, triangles: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    计算插值系数 a_i

    a_i = x_j * y_m - x_m * y_j

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)，其中 n 是点的数量。
        triangles (NDArray): 三角形的顶点索引，形状为 (m, 3)，其中 m 是三角形的数量。

    Returns:
        tuple[NDArray, NDArray, NDArray]: 返回插值系数 a_i 的数组，元组对应i,j,m。
    """
    assert points.shape[1] == 2, "points must be of shape (n, 2)"
    assert triangles.shape[1] == 3, "triangles must be of shape (m, 3)"

    points_triangles = points[triangles]

    x_i: NDArray = points_triangles[:, 0, 0]
    y_i: NDArray = points_triangles[:, 0, 1]
    x_j: NDArray = points_triangles[:, 1, 0]
    y_j: NDArray = points_triangles[:, 1, 1]
    x_m: NDArray = points_triangles[:, 2, 0]
    y_m: NDArray = points_triangles[:, 2, 1]

    a_i: NDArray = x_j * y_m - x_m * y_j
    a_j: NDArray = x_i * y_m - x_m * y_i
    a_m: NDArray = x_i * y_j - x_j * y_i

    return a_i, a_j, a_m


def b_list(points: NDArray, triangles: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    计算插值系数 b_i

    b_i = y_j - y_m

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)，其中 n 是点的数量。
        triangles (NDArray): 三角形的顶点索引，形状为 (m, 3)，其中 m 是三角形的数量。

    Returns:
        tuple[NDArray, NDArray, NDArray]: 返回插值系数 b_i 的数组，元组对应i,j,m。
    """
    assert points.shape[1] == 2, "points must be of shape (n, 2)"
    assert triangles.shape[1] == 3, "triangles must be of shape (m, 3)"

    points_triangles = points[triangles]

    y_i: NDArray = points_triangles[:, 0, 1]
    y_j: NDArray = points_triangles[:, 1, 1]
    y_m: NDArray = points_triangles[:, 2, 1]

    b_i: NDArray = y_j - y_m
    b_j: NDArray = y_i - y_m
    b_m: NDArray = y_i - y_j

    return b_i, b_j, b_m


def c_list(points: NDArray, triangles: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    计算插值系数 c_i

    c_i = x_m - x_j

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)，其中 n 是点的数量。
        triangles (NDArray): 三角形的顶点索引，形状为 (m, 3)，其中 m 是三角形的数量。

    Returns:
        tuple[NDArray, NDArray, NDArray]: 返回插值系数 c_i 的数组，元组对应i,j,m。
    """
    assert points.shape[1] == 2, "points must be of shape (n, 2)"
    assert triangles.shape[1] == 3, "triangles must be of shape (m, 3)"

    points_triangles = points[triangles]

    x_i: NDArray = points_triangles[:, 0, 0]
    x_j: NDArray = points_triangles[:, 1, 0]
    x_m: NDArray = points_triangles[:, 2, 0]

    c_i: NDArray = x_m - x_j
    c_j: NDArray = x_m - x_i
    c_m: NDArray = x_j - x_i

    return c_i, c_j, c_m


def get_K_rs(
    points: NDArray,
    triangles: NDArray,
    b_list: tuple[NDArray, NDArray, NDArray],
    c_list: tuple[NDArray, NDArray, NDArray],
    r: int,
    s: int,
    E: float = 1.0,
    nu: float = 0.3,
    t: float = 1.0,
) -> NDArray:
    """
    计算单元刚度阵分块阵 K_rs

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)，其中 n 是点的数量。
        triangles (NDArray): 三角形的顶点索引，形状为 (m, 3)，其中 m 是三角形的数量。
        b_list (tuple[NDArray, NDArray, NDArray]): 插值系数 b_i 的元组。
        c_list (tuple[NDArray, NDArray, NDArray]): 插值系数 c_i 的元组。
        r (int): 行索引。
        s (int): 列索引。
        E (float): 弹性模量，默认为 1.0。
        nu (float): 泊松比，默认为 0.3。
        t (float): 厚度，默认为 1.0。

    Returns:
        NDArray: 返回单元刚度阵分块阵 K_rs。
    """

    K_1: NDArray = b_list[r] * b_list[s] + (1 - nu) * c_list[r] * c_list[s] / 2
    K_2: NDArray = nu * b_list[s] * c_list[r] + (1 - nu) * b_list[r] * c_list[s] / 2
    K_3: NDArray = nu * b_list[r] * c_list[s] + (1 - nu) * b_list[s] * c_list[r] / 2
    K_4: NDArray = c_list[r] * c_list[s] + (1 - nu) * b_list[r] * b_list[s] / 2

    area: NDArray = tri_area(points[triangles])
    K_rs: NDArray = (E * t / ((1 - nu**2) * 4 * area)).reshape(
        (-1, 1, 1)
    ) * np.concatenate(
        (
            np.column_stack((K_1, K_2)).reshape((-1, 2, 1)),
            np.column_stack((K_3, K_4)).reshape((-1, 2, 1)),
        ),
        axis=2,
    )

    return K_rs

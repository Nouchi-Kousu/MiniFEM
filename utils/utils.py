import numpy as np
from numpy.typing import NDArray


def pt_in_tri(
    points: NDArray, triangles: NDArray, save_shape: bool = False
) -> NDArray:
    """
    使用重心法判断点是否在三角形内, 允许基数为 1 时的广播

    https://www.cnblogs.com/graphics/archive/2010/08/05/1793393.html

    Args:
        point (NDArray): 用于判断的点, shape (n,2), n 与三角形个数对应.
        triangles (NDArray): 用于判断的三角形, shape (n, 3, 2), n 是三角形个数.
        save_shape (bool): 是否保存向量的形状.

    Returns:
        NDArray: 返回 bool 类型的NumPy张量, 形状由 save_shape 决定.
    """
    assert points.shape[1] == 2, "点张量形状应为 (n, 2)"
    assert triangles.shape[1] == 3 and triangles.shape[2] == 2, (
        "三角形张量形状应为 (n, 3, 2)"
    )
    assert (
        points.shape[0] == triangles.shape[0]
        or points.shape[0] == 1
        or triangles.shape[0] == 1
    ), "点和三角形的元素数量必须相同, 或者其中一个为 1"

    # 处理点和三角形的广播
    if points.shape[0] == 1:
        points = np.tile(points, (triangles.shape[0], 1))
    if triangles.shape[0] == 1:
        triangles = np.tile(triangles, (points.shape[0], 1, 1))

    # 计算三角形的边向量与目标点向量
    v0: NDArray = triangles[:, 1, :] - triangles[:, 0, :]
    v1: NDArray = triangles[:, 2, :] - triangles[:, 0, :]
    v2: NDArray = points - triangles[:, 0, :]

    # 计算向量点积
    d00: NDArray = np.einsum("ij,ij->i", v0, v0)
    d01: NDArray = np.einsum("ij,ij->i", v0, v1)
    d11: NDArray = np.einsum("ij,ij->i", v1, v1)
    d20: NDArray = np.einsum("ij,ij->i", v2, v0)
    d21: NDArray = np.einsum("ij,ij->i", v2, v1)

    # 点坐标计算
    denom: NDArray = d00 * d11 - d01 * d01
    u: NDArray = (d11 * d20 - d01 * d21) / denom
    v: NDArray = (d00 * d21 - d01 * d20) / denom

    if save_shape:
        return (
            np.logical_and(u >= 0, v >= 0) & np.logical_and(u + v <= 1, u + v >= 0)
        ).reshape(points.shape[0], 1)
    else:
        return (
            np.logical_and(u >= 0, v >= 0) & np.logical_and(u + v <= 1, u + v >= 0)
        ).reshape(-1)


def tri_centroid(triangles: NDArray) -> NDArray:
    """
    计算三角形的重心

    Args:
        triangles (NDArray): 由三角形顶点坐标组成的数组，形状为 (n, 3, 2)

    Returns:
        NDArray: 三角形的重心坐标，形状为 (n, 2)
    """
    assert triangles.shape[1:] == (3, 2), "三角形顶点编号数组的形状应为 (n, 3, 2)"

    # 计算重心坐标
    center_of_gravity = np.mean(triangles, axis=1)
    return center_of_gravity


def tri_area(triangles: NDArray, save_shape: bool = False) -> NDArray:
    """
    计算三角形的面积

    Args:
        triangles (NDArray): 由三角形顶点坐标组成的数组，形状为 (n, 3, 2)
        save_shape (bool): 是否保存向量的形状.
        如果为 True，则返回的面积形状为 (n, 1)，否则为 (n,)

    Returns:
        NDArray: 三角形的面积，形状为 (n,) 或 (n, 1)
    """
    assert triangles.shape[1:] == (3, 2), "三角形顶点编号数组的形状应为 (n, 3, 2)"

    # 计算三角形的面积
    area: NDArray = (
        np.abs(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )
        )
        / 2
    )
    return area.reshape(-1, 1) if save_shape else area


def pt2tri_max_dist(
    points: NDArray, triangles: NDArray, save_shape: bool = False
) -> NDArray:
    """
    计算点到三角形的最大距离

    Args:
        points (NDArray): 点坐标，形状为 (n, 2)
        triangles (NDArray): 三角形顶点坐标，形状为 (n, 3, 2)
        save_shape (bool): 是否保存向量的形状.

    Returns:
        NDArray: 点到三角形的最大距离，形状为 (n,) 或 (n, 1)
    """
    assert points.shape[1] == 2, "点张量形状应为 (n, 2)"
    assert triangles.shape[1:] == (3, 2), "三角形顶点编号数组的形状应为 (n, 3, 2)"
    assert (
        points.shape[0] == triangles.shape[0]
        or points.shape[0] == 1
        or triangles.shape[0] == 1
    ), "点和三角形的元素数量必须相同, 或者其中一个为 1"

    # 计算点到三角形的最大距离
    max_distance: NDArray = np.max(
        np.linalg.norm(points[:, np.newaxis] - triangles, axis=2), axis=1
    )
    return max_distance.reshape(-1, 1) if save_shape else max_distance

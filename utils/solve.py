import numpy as np
from numpy.typing import NDArray
from utils.utils import tri_area
from scipy.sparse import lil_array, csr_array


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


def assemble_global_stiffness(
    points: NDArray, triangles: NDArray, E: float = 1.0, nu: float = 0.3, t: float = 1.0
) -> csr_array:
    """
    为给定的点和三角形计算全局刚度矩阵。

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)，其中 n 是点的数量。
        triangles (NDArray): 三角形的顶点索引，形状为 (m, 3)，其中 m 是三角形的数量。
        E (float): 弹性模量，默认为 1.0。
        nu (float): 泊松比，默认为 0.3。
        t (float): 厚度，默认为 1.0。

    Returns:
        csr_array: 返回csr_matrix格式的全局刚度矩阵的压缩稀疏行格式。
    """
    K_e: list[list[NDArray]] = [
        [
            get_K_rs(
                points,
                triangles,
                b_list(points, triangles),
                c_list(points, triangles),
                i,
                j,
                E=E,
                nu=nu,
                t=t,
            )
            for j in range(3)
        ]
        for i in range(3)
    ]

    K_global: lil_array = lil_array((len(points) * 2, len(points) * 2), dtype=float)

    for idx, tri in enumerate(triangles):
        for i in range(3):
            for j in range(3):
                K_global[tri[i] * 2, tri[j] * 2] += K_e[i][j][idx, 0, 0]
                K_global[tri[i] * 2, tri[j] * 2 + 1] += K_e[i][j][idx, 0, 1]
                K_global[tri[i] * 2 + 1, tri[j] * 2] += K_e[i][j][idx, 1, 0]
                K_global[tri[i] * 2 + 1, tri[j] * 2 + 1] += K_e[i][j][idx, 1, 1]

    return K_global.tocsr()


def enforce_constraints_diagonal_fix(
    constraints_list: tuple[list[int], list[float]],
    K_global: csr_array,
    P_global: NDArray,
) -> tuple[csr_array, NDArray]:
    """
    使用对角元乘大数法对结构施加约束。

    Args:
        constraints_list (tuple[list[int], list[float]]): 约束的节点索引列表和对应的约束列表。
        K_global (csr_array): 全局刚度矩阵。
        P_global (NDArray): 全局载荷向量。

    Returns:
        tuple[csr_array, NDArray]: 返回施加约束后的全局刚度矩阵和载荷向量。
    """

    for dof, val in zip(constraints_list[0], constraints_list[1]):
        K_global[dof, dof] *= 1e10
        P_global[dof] = K_global[dof, dof] * val

    return K_global, P_global


def P_global(load_tuple: tuple[list[int], list[list[float]]]) -> NDArray:
    """
    生成全局载荷向量。

    Args:
        load_tuple (tuple[list[int], list[float]]): 载荷的节点索引列表和对应的载荷列表。

    Returns:
        NDArray: 返回全局载荷向量。
    """
    P_global: NDArray = np.zeros(len(load_tuple[0]) * 2, dtype=float)

    for dof, val in zip(load_tuple[0], load_tuple[1]):
        P_global[dof * 2] = val[0]
        P_global[dof * 2 + 1] = val[1]

    return P_global

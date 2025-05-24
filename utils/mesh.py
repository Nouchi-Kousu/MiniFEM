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

    # 计算外接圆的圆心和半径
    ab = b - a
    ac = c - a
    ab_mid = (a + b) / 2
    ac_mid = (a + c) / 2

    # 计算垂直平分线的方向向量
    ab_perp = np.array([-ab[:, 1], ab[:, 0]]).T
    ac_perp = np.array([-ac[:, 1], ac[:, 0]]).T

    # 初始化结果数组
    is_in_circle = np.zeros(triangle.shape[0], dtype=bool)

    # 对每个三角形单独计算外接圆圆心和判断点是否在圆内
    for i in range(triangle.shape[0]):
        current_ab_perp = ab_perp[i]
        current_ac_perp = ac_perp[i]

        current_ab_mid = ab_mid[i]
        current_ac_mid = ac_mid[i]

        # 构成线性方程组的矩阵 A
        # A = [[ab_perp_x, ab_perp_y], [ac_perp_x, ac_perp_y]]
        # 行列式 det(A) = ab_perp_x * ac_perp_y - ab_perp_y * ac_perp_x
        # 这等价于向量 ab 和 ac 的2D叉积 (ab_x * ac_y - ab_y * ac_x)
        # 如果为0，则 ab 和 ac 平行，意味着原始点 a, b, c 共线
        A_matrix = np.stack([current_ab_perp, current_ac_perp], axis=0)

        # 检查矩阵是否奇异 (即顶点是否共线)
        # 使用一个小的容差值 epsilon 来比较浮点数
        if np.abs(np.linalg.det(A_matrix)) < 1e-12:
            is_in_circle[i] = False  # 对于退化三角形，点不在其外接圆内
        else:
            # 线性方程组的右侧向量 b
            b_vector = current_ac_mid - current_ab_mid  # This is dm

            # 当前边的垂直平分线向量
            v1 = current_ab_perp
            v2 = current_ac_perp

            # 我们要求解 t * v1 - s * v2 = b_vector 得到 t 和 s
            # 对应的矩阵 M_solve = [[v1_x, -v2_x], [v1_y, -v2_y]]
            M_solve = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
            try:
                # 解出参数 t 和 s
                params = np.linalg.solve(M_solve, b_vector)
                t_param = params[0]

                # 计算外接圆心
                current_circumcenter = current_ab_mid + t_param * v1

                # 当前三角形的顶点 a
                current_a_vertex = a[i]

                # 计算外接圆半径
                current_circumradius = np.linalg.norm(
                    current_circumcenter - current_a_vertex
                )
                # 计算点到圆心的距离
                current_point_dist = np.linalg.norm(
                    current_circumcenter - point
                )  # 'point' 是待检查的点

                is_in_circle[i] = current_point_dist < current_circumradius
            except np.linalg.LinAlgError:
                # 理论上行列式检查应该能捕获此情况，这里作为备用
                is_in_circle[i] = False

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

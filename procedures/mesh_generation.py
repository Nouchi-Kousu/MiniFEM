from re import A
import numpy as np
import utils.utils as utils
import utils.mesh as mesh
from numpy.typing import NDArray
from utils.utils import TriangleArray, PointArray
from utils.plot import plot_tri
import matplotlib.pyplot as plt


def insert_pts_into_mesh(
    points: PointArray, triangles: TriangleArray, insert_points: NDArray
) -> tuple[PointArray, TriangleArray]:
    """
    将点插入到已有三角网格中

    使用Bowyer-Watson增量式算法

    Args:
        points (PointArray): 点的坐标，形状为 (n, 2)
        triangles (TriangleArray): 三角形的索引，形状为 (m, 3)
        insert_points (NDArray): 要插入的点，形状为 (k, 2)

    Returns:
        tuple[PointArray, TriangleArray]: 插入后的三角形的顶点坐标和三角形的索引
    """
    assert insert_points.shape[1] == 2, "增量点张量形状必须为 (k, 2)"

    for i in range(insert_points.shape[0]):
        # tri = triangles.to_numpy()[~triangles.get_del()]
        # plot_tri(points.to_numpy(), triangles.to_numpy()[~triangles.get_del()])
        # plt.show()

        innsert_point = insert_points[i]
        # 1. 寻找需要删除的星形多边形内三角形
        triangles_numpy = triangles.to_numpy()
        points_nmupy = points.to_numpy()
        del_bool = mesh.pt_in_circle(innsert_point, triangles_numpy, points_nmupy)
        # 只保留有效三角形网格
        del_bool = del_bool & (~triangles.get_del())
        del_tri = triangles_numpy[del_bool]
        triangles.del_tri(del_bool)
        # 构建星形多边形
        tris_unique_edges = mesh.tris_unique_edges(del_tri)
        point_index: int = points.shape()[0]
        points.add(innsert_point)
        new_tri = np.zeros((tris_unique_edges.shape[0], 3), dtype=int)
        new_tri[:, :2] = tris_unique_edges
        new_tri[:, 2] = point_index
        new_tri.sort()

        triangles.add(new_tri)

        triangles.clean_up()

    return points, triangles


def gen_mesh_from_pts(points: NDArray) -> tuple[NDArray, NDArray]:
    """
    从给定点生成三角形网格

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)

    Returns:
        tuple[NDArray, NDArray]: 三角形的顶点坐标和三角形的索引
    """
    right_top_pt: NDArray = np.max(points, axis=0)
    left_button_pt: NDArray = np.min(points, axis=0)
    width: float = right_top_pt[0] - left_button_pt[0]
    height: float = right_top_pt[1] - left_button_pt[1]

    left: float = left_button_pt[0] - width * 3
    button: float = left_button_pt[1] - height * 3
    right: float = right_top_pt[0] + width * 3
    top: float = right_top_pt[1] + height * 3

    super_points: NDArray = np.array(
        [[left, button], [right, button], [right, top], [left, top]]
    )
    super_tri: NDArray = np.array([[0, 1, 3], [1, 2, 3]])

    pt_tmp = PointArray()
    pt_tmp.add(super_points)
    tri_tmp = TriangleArray()
    tri_tmp.add(super_tri)

    pt_all, tri_all = insert_pts_into_mesh(pt_tmp, tri_tmp, points)

    pt_return = pt_all.to_numpy()[4:]
    tri_return = tri_all.to_numpy()[~tri_all.get_del()]
    index = tri_return[:, 0] != 0
    index = index & (tri_return[:, 0] != 1)
    index = index & (tri_return[:, 0] != 2)
    index = index & (tri_return[:, 0] != 3)
    tri_return = tri_return[index]

    return pt_return, tri_return - 4


def refine_mesh(
    points: NDArray, triangles: NDArray, size: float
) -> tuple[NDArray, NDArray]:
    """
    对三角形网格进行细化
    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)
        triangles (NDArray): 三角形的索引，形状为 (m, 3)
        size (float): 网格尺寸

    Returns:
        tuple[NDArray, NDArray]: 细化后的三角形的顶点坐标和三角形的索引
    """
    assert points.shape[1] == 2, "点的坐标形状必须为 (n, 2)"
    assert triangles.shape[1] == 3, "三角形的索引形状必须为 (m, 3)"

    pt_tmp = PointArray()
    pt_tmp.add(points)
    tri_tmp = TriangleArray()
    tri_tmp.add(triangles)

    def refine_step(pt_tmp, tri_tmp) -> NDArray:
        centroids = utils.tri_centroid(pt_tmp.to_numpy()[tri_tmp.to_numpy()])
        max_dist = utils.pt2tri_max_dist(
            centroids, pt_tmp.to_numpy()[tri_tmp.to_numpy()]
        )
        size_bool = max_dist > size
        size_bool = size_bool & (~tri_tmp.get_del())
        centroids = centroids[size_bool]
        return centroids

    centroids = refine_step(pt_tmp, tri_tmp)

    while centroids.shape[0] > 0:
        print(
            f"细化中，当前剩余点数: {centroids.shape[0]}, 当前网格点数: {pt_tmp.shape()[0]}"
        )
        insert_pts_into_mesh(pt_tmp, tri_tmp, centroids)
        centroids = refine_step(pt_tmp, tri_tmp)

        tri_tmp.clean_up()

    return pt_tmp.to_numpy(), tri_tmp.to_numpy()[~tri_tmp.get_del()]


def sample_pts_on_edges(points: NDArray, adj: NDArray, size: float) -> NDArray:
    """
    在结构边上均匀布种

    Args:
        points (NDArray): 点的坐标，形状为 (n, 2)
        triangles (NDArray): 结构边邻接列表，形状为 (m, 2)
        size (float): 网格尺寸
    Returns:
        NDArray: 采样点的坐标，形状为 (k, 2)
    """
    assert points.shape[1] == 2, "点的坐标形状必须为 (n, 2)"
    assert adj.shape[1] == 2, "结构边邻接列表形状必须为 (m, 2)"

    num_list: NDArray = (
        np.linalg.norm(points[adj][:, 0] - points[adj][:, 1], axis=1) // size
    )

    num_list = num_list.astype(int)

    seeds = np.zeros((num_list.sum(), 2), dtype=float)

    j, k = 0, 0

    for i in range(num_list.shape[0]):
        j = k
        k += num_list[i]
        pt1 = points[adj[i, 0]]
        pt2 = points[adj[i, 1]]
        seeds[j:k, 0] = np.linspace(pt1[0], pt2[0], num_list[i] + 2)[1:-1]
        seeds[j:k, 1] = np.linspace(pt1[1], pt2[1], num_list[i] + 2)[1:-1]

    return np.concatenate((points, seeds), axis=0)

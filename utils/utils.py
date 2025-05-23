import numpy as np
from numpy.typing import NDArray


def pt_in_tri(points: NDArray, triangles: NDArray, save_shape: bool = False) -> NDArray:
    """
    使用重心法判断点是否在三角形内, 允许基数为 1 时的广播

    https://www.cnblogs.com/graphics/archive/2010/08/05/1793393.html

    Args:
        point (NDArray): 用于判断的点，形状为 (n,2), n 与三角形个数对应.
        triangles (NDArray): 用于判断的三角形，形状为 (n, 3, 2), n 是三角形个数.
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


def expand_array(array, new_size):
    """
    扩展 numpy 数组到指定大小。
    """
    new_array: NDArray = np.zeros((new_size, *array.shape[1:]), dtype=array.dtype)
    new_array[: array.shape[0]] = array
    return new_array


class PointArray:
    def __init__(self, initial_capacity=100):
        """
        初始化点数组。

        Args:
            initial_capacity (int): 初始点容量。
        """
        self._data = np.zeros((initial_capacity, 2), dtype=float)
        self._size = 0
        self._capacity = initial_capacity

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return self._data[: self._size][index]

    def __setitem__(self, index, value):
        assert value.shape == (2,), "点的形状必须为 (2,)"
        if index >= self._size:
            raise IndexError("只能修改已存在的点")
        self._data[index] = value

    def add(self, point: NDArray):
        """
        添加一个点。

        Args:
            point (NDArray): 点的坐标, shape (2,)
        """
        assert point.shape == (2,), "点的形状必须为 (2,)"
        if self._size >= self._capacity:
            self._expand()
        self._data[self._size] = point
        self._size += 1

    def _expand(self):
        """
        扩展点数组的容量。
        """
        new_capacity = self._capacity * 2
        new_data = np.zeros((new_capacity, 2), dtype=float)
        new_data[: self._size] = self._data
        self._data = new_data
        self._capacity = new_capacity

    def to_numpy(self) -> NDArray:
        """
        返回有效点的 NumPy 数组。

        Returns:
            NDArray: 有效点的数组，形状为 (n, 2)
        """
        return self._data[: self._size]


class TriangleArray:
    def __init__(self, initial_capacity=100):
        """
        初始化三角形数组。

        Args:
            initial_capacity (int): 初始三角形容量。
        """
        self._data = np.zeros((initial_capacity, 3), dtype=int)
        self._size = 0
        self._capacity = initial_capacity
        self._is_del = np.zeros((initial_capacity, 1), dtype=bool)
        self._tri_num = 0

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return self._data[: self._size][index]

    def __setitem__(self, index, value):
        assert value.shape == (3,), "三角形的形状必须为 (3,)"
        if index >= self._size:
            raise IndexError("只能修改已存在的三角形")
        self._data[index] = value

    def add(self, triangle: NDArray):
        """
        添加一个三角形。

        Args:
            triangle (NDArray): 三角形的顶点索引，形状为 (3,) 或 (n, 3)
        """
        assert triangle.shape == (3,) or triangle.shape[1] == 3, (
            "三角形的形状必须为 (3,) 或 (n, 3)"
        )
        if self._size >= self._capacity:
            self._expand()
        if triangle.shape[0] == 3:
            self._data[self._size] = triangle
            self._size += 1
            self._tri_num += 1
        elif n := triangle.shape[0] > 3:
            self._data[self._size : self._size + n] = triangle
            self._size += n
            self._tri_num += n

    def _expand(self):
        """
        扩展三角形数组的容量。
        """
        new_capacity = self._capacity * 2
        new_data = np.zeros((new_capacity, 3), dtype=int)
        new_data[: self._size] = self._data
        self._data = new_data
        self._capacity = new_capacity

    def to_numpy(self) -> NDArray:
        """
        返回有效三角形的 NumPy 数组。

        Returns:
            NDArray: 有效三角形的数组，形状为 (n, 3)
        """
        return self._data[: self._size]

    def del_tri(self, index: int | list) -> None:
        """
        删除指定索引的三角形。

        Args:
            index (int | list): 要删除的三角形索引。
        """

        self._is_del[index] = True
        self._tri_num -= 1 if isinstance(index, int) else len(index)

    def get_tri_num(self) -> int:
        """
        获取当前有效三角形的数量。

        Returns:
            int: 有效三角形的数量。
        """
        return self._tri_num

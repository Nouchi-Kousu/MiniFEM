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
        添加点。

        Args:
            point (NDArray): 点的坐标，形状为 (2,) 或 (n, 2)
        """
        assert point.shape == (2,) or point.shape[1] == 2, (
            "点的形状必须为 (2,) 或 (n, 2)"
        )
        if point.shape == (2,):  # Single point
            if (
                self._size + 1 > self._data.shape[0]
            ):  # Use > to expand when exactly full and adding one more
                self._expand()
            self._data[self._size] = point
            self._size += 1
        else:  # Multiple points
            num_to_add = point.shape[0]
            while self._size + num_to_add > self._data.shape[0]:
                self._expand()
            self._data[self._size : self._size + num_to_add] = point
            self._size += num_to_add

    def _expand(self):
        """
        扩展点数组的容量。
        """
        current_allocated_size = self._data.shape[0]
        if current_allocated_size == 0:
            new_allocated_size = (
                10  # Default minimum capacity if array was initially empty
            )
        else:
            new_allocated_size = current_allocated_size * 2

        new_data = np.zeros((new_allocated_size, 2), dtype=float)
        if self._size > 0:  # Copy existing data if any
            new_data[: self._size] = self._data[: self._size]
        self._data = new_data

    def to_numpy(self) -> NDArray:
        """
        返回有效点的 NumPy 数组。

        Returns:
            NDArray: 有效点的数组，形状为 (n, 2)
        """
        return self._data[: self._size]

    def shape(self) -> tuple:
        """
        获取点数组的形状。

        Returns:
            tuple[int, int]: 点数组的形状。
        """
        return self.to_numpy().shape


class TriangleArray:
    def __init__(self, initial_capacity=100):
        """
        初始化三角形数组。

        Args:
            initial_capacity (int): 初始三角形容量。
        """
        self._data = np.zeros((initial_capacity, 3), dtype=int)
        self._size = 0
        self._is_del = np.zeros((initial_capacity), dtype=bool)
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

        if triangle.ndim == 1 and triangle.shape[0] == 3:  # Single triangle
            num_to_add = 1
            if self._size + num_to_add > self._data.shape[0]:
                self._expand()
            self._data[self._size] = triangle
            self._is_del[self._size] = (
                False  # Ensure new triangle is not marked as deleted
            )
            self._size += num_to_add
            self._tri_num += num_to_add
        elif triangle.ndim == 2 and triangle.shape[1] == 3:  # Multiple triangles
            num_to_add = triangle.shape[0]
            if num_to_add == 0:  # Nothing to add
                return
            while self._size + num_to_add > self._data.shape[0]:
                self._expand()
            self._data[self._size : self._size + num_to_add] = triangle
            self._is_del[self._size : self._size + num_to_add] = (
                False  # Ensure new triangles are not marked deleted
            )
            self._size += num_to_add
            self._tri_num += num_to_add
        else:
            # Should be caught by assert, but as a fallback
            raise ValueError("Invalid triangle shape")

    def _expand(self):
        """
        扩展三角形数组的容量。
        """
        current_allocated_size = self._data.shape[0]
        if current_allocated_size == 0:
            new_allocated_size = 10  # Default minimum capacity
        else:
            new_allocated_size = current_allocated_size * 2

        new_data = np.zeros((new_allocated_size, 3), dtype=int)
        new_is_del = np.zeros((new_allocated_size), dtype=bool)

        if self._size > 0:  # Copy existing data if any
            new_data[: self._size] = self._data[: self._size]
            new_is_del[: self._size] = self._is_del[: self._size]

        self._data = new_data
        self._is_del = new_is_del

    def to_numpy(self) -> NDArray:
        """
        返回有效三角形的 NumPy 数组。

        Returns:
            NDArray: 有效三角形的数组，形状为 (n, 3)
        """
        return self._data[: self._size]

    def del_tri(self, index: int | list | NDArray) -> None:
        """
        删除指定索引的三角形。

        Args:
            index (int | list | NDArray): 要删除的三角形索引。NDArray 为布尔索引
        """
        if isinstance(index, int):
            if not self._is_del[index]:
                self._is_del[index] = True
                self._tri_num -= 1
        elif isinstance(index, list):
            index = np.array(index, dtype=int)
            valid_deletions = ~self._is_del[index]
            self._is_del[index] = True
            self._tri_num -= np.sum(valid_deletions)
        elif isinstance(index, np.ndarray):
            if index.dtype == bool:
                index_all = np.zeros_like(self._is_del, dtype=bool)
                index_all[: index.shape[0]] = index
                valid_deletions = ~self._is_del[index_all]
                self._is_del[index_all] = True
                self._tri_num -= np.sum(valid_deletions)
            else:
                valid_deletions = ~self._is_del[index]
                self._is_del[index] = True
                self._tri_num -= np.sum(valid_deletions)

    def get_tri_num(self) -> int:
        """
        获取当前有效三角形的数量。

        Returns:
            int: 有效三角形的数量。
        """
        return self._tri_num

    def shape(self) -> tuple:
        """
        获取三角形数组的形状。

        Returns:
            tuple[int, int]: 三角形数组的形状。
        """
        return self.to_numpy().shape

    def get_del(self) -> NDArray:
        return self._is_del[: self._size]

    def clean_up(self) -> None:
        """
        清理被删除的三角形，减小存储的三角形数量。
        """
        if not np.any(self._is_del[: self._size]):
            # 没有需要清理的三角形
            return

        valid_triangles_mask = ~self._is_del[: self._size]
        self._data[: self._tri_num] = self._data[: self._size][valid_triangles_mask]
        # For the compacted part, reset deletion flags
        self._is_del[: self._tri_num] = False
        # Any flags beyond _tri_num in the _is_del array are irrelevant as _size will be _tri_num

        self._size = self._tri_num
        # The commented out block for resizing _capacity, _data, and _is_del is removed
        # as _capacity is removed. Actual shrinking of np arrays is a separate step if desired.

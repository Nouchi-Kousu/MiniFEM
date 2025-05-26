import numpy as np
from numpy.typing import NDArray


def distribute_edge_loads(
    points: NDArray,
    adj: NDArray,
    index_list: list[int],
    load_edge: int | list[int],
    load: list[float] | list[list[float]],
) -> tuple[list[int], list[list[float]]]:
    """
    对选择边施加均布载荷转化为等效节点力

    默认节点在边上均匀分布

    Args:
        points (NDArray): 多边形定义点的坐标，形状为 (n, 2)
        adj (NDArray): 多边形边的索引，形状为 (m, 2)
        index_list (list[int]): 边对应构成节点的索引列表
        load_edge (int | list[int]): 施加载荷的边索引或边索引列表
        load (list[float] | list[list[float]]): 施加在边上的载荷,
            如果是单一载荷，则为一个列表，形状为 (2, )，对应均布载荷的x和y分量；
            如果是多个载荷，则为一个 (2, ) 列表的列表，长度应与边索引列表相同，其中每个列表均为均布载荷的x和y分量

    Returns:
        tuple[list[int], list[list[float]]]: 返回两个列表，第一个是施加载荷的节点索引列表，
            第二个是每个节点对应的等效载荷列表，形状为 (n, 2)，其中 n 是施加载荷的节点数
    """
    assert points.shape[1] == 2, "点的坐标形状必须为 (n, 2)"
    assert adj.shape[1] == 2, "边的索引形状必须为 (m, 2)"
    assert len(index_list) == adj.shape[0] + 1, "index_list 长度必须与边的数量相同"

    load_edge = [load_edge] if isinstance(load_edge, int) else load_edge

    load = [load] * len(load_edge) if isinstance(load[0], float) else load  # type: ignore

    assert len(load_edge) == len(load), "load_edge 和 load 的长度必须相同"

    node_list: list[int] = []
    node_loads: list[list[float]] = []

    for idx in range(len(load_edge)):
        adj_idx: int = load_edge[idx]
        node_list += list(
            range(
                index_list[adj_idx] + points.shape[0],
                index_list[adj_idx + 1] + points.shape[0],
            )
        )
        node_list += [
            int(adj[adj_idx, 0]),
            int(adj[adj_idx, 1]),
        ]
        node_num: int = index_list[adj_idx + 1] - index_list[adj_idx]
        edge_len: float = float(
            np.linalg.norm(points[adj[adj_idx, 0]] - points[adj[adj_idx, 1]])
        )
        load_x: float = load[idx][0] * edge_len / (node_num + 1)  # type: ignore
        load_y: float = load[idx][1] * edge_len / (node_num + 1)  # type: ignore
        node_loads += [[load_x, load_y]] * node_num
        node_loads += [[load_x / 2, load_y / 2]] * 2

    return node_list, node_loads


def apply_edge_constraints(
    points: NDArray,
    adj: NDArray,
    index_list: list[int],
    constraints_edge: int | list[int],
    constraints: list[bool] | list[list[bool]],
) -> tuple[list[int], list[list[bool]]]:
    """
    对选择边施加均布载荷转化为等效节点力

    默认节点在边上均匀分布

    Args:
        points (NDArray): 多边形定义点的坐标，形状为 (n, 2)
        adj (NDArray): 多边形边的索引，形状为 (m, 2)
        index_list (list[int]): 边对应构成节点的索引列表
        load_edge (int | list[int]): 施加约束的边索引或边索引列表
        load (list[bool] | list[list[bool]]): 施加在边上的载荷,
            如果是单一载荷，则为一个列表，形状为 (2, )，对应是否对节点施加x和y方向的约束；
            如果是多个载荷，则为一个 (2, ) 列表的列表，长度应与边索引列表相同，其中每个列表均为是否对节点施加x和y方向的约束

    Returns:
        tuple[list[int], list[list[bool]]]: 返回两个列表，第一个是施加约束的节点索引列表，
            第二个是每个节点对应的等效载荷列表，形状为 (n, 2)，其中 n 是施加约束的节点数
    """
    assert points.shape[1] == 2, "点的坐标形状必须为 (n, 2)"
    assert adj.shape[1] == 2, "边的索引形状必须为 (m, 2)"
    assert len(index_list) == adj.shape[0] + 1, "index_list 长度必须与边的数量相同"

    constraints_edge = (
        [constraints_edge] if isinstance(constraints_edge, int) else constraints_edge
    )

    constraints = (
        [constraints] * len(constraints_edge)
        if isinstance(constraints[0], bool)
        else constraints
    )  # type: ignore

    assert len(constraints_edge) == len(constraints), (
        "constraints_edge 和 constraints 的长度必须相同"
    )

    node_list: list[int] = []
    node_loads: list[list[bool]] = []

    for idx in range(len(constraints_edge)):
        adj_idx: int = constraints_edge[idx]
        node_list += list(
            range(
                index_list[adj_idx] + points.shape[0],
                index_list[adj_idx + 1] + points.shape[0],
            )
        )
        node_list += [
            int(adj[adj_idx, 0]),
            int(adj[adj_idx, 1]),
        ]
        node_num: int = index_list[adj_idx + 1] - index_list[adj_idx]

        load_x: bool = constraints[idx][0] # type: ignore
        load_y: bool = constraints[idx][1] # type: ignore

        node_loads += [[load_x, load_y]] * node_num
        node_loads += [[load_x, load_y]] * 2

    return node_list, node_loads

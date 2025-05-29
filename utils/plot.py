import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.colors import Normalize

from numpy.typing import NDArray
import numpy as np
from rich.console import Console

console = Console()


def plot_tri(
    points: NDArray,
    triangles: NDArray,
    stresses: NDArray | None = None,
    color: str = "#66ccff",
    linewidth: float = 0.5,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    绘制划分好的三角形网格，或绘制带应力的三角形网格

    Args:
        triangles(NDArray): 由三角形顶点编号组成的数组，形状为 (n, 3)
        points(NDArray): 三角形顶点坐标，形状为 (m, 2)
        stresses(NDArray | None): 每个三角形的应力值数组，形状为 (n, )，可选
        color(str): 当没有应力表示时，三角形的填充颜色

    Returns:
        fig, ax (tuple): 返回一个包含两个元素的元组，第一个元素是绘制的图形对象，第二个元素是坐标轴对象

    """
    console.log("[blue]绘制三角形网格[/blue]")
    assert triangles.shape[1] == 3, "三角形顶点编号数组的形状应为 (n, 3)"
    assert points.shape[1] == 2, "三角形顶点坐标数组的形状应为 (m, 2)"
    if stresses is not None:
        assert stresses.shape[0] == triangles.shape[0], (
            "应力值数组的长度应与三角形数组的长度一致"
        )

    fig, ax = plt.subplots()

    if stresses is not None:
        # 使用colormap绘制应力云图
        norm: Normalize = Normalize(vmin=np.min(stresses), vmax=np.max(stresses))
        # 使用rainbow colormap表征应力状况
        cmap = cm.get_cmap("rainbow")

        for tri, stress in zip(triangles, stresses):
            pts: NDArray = points[tri]
            ax.add_patch(
                Polygon(
                    pts,
                    edgecolor="k",
                    fill=True,
                    facecolor=cmap(norm(stress)),
                    linewidth=linewidth,
                    **kwargs,
                )
            )
    else:
        # 使用单一颜色绘制三角形
        for tri in triangles:
            pts: NDArray = points[tri]
            ax.add_patch(
                Polygon(
                    pts,
                    edgecolor="k",
                    fill=True,
                    facecolor=color,
                    linewidth=linewidth,
                    **kwargs,
                )
            )

    # 设置坐标轴范围
    width: float = np.max(points[:, 0]) - np.min(points[:, 0])
    height: float = np.max(points[:, 1]) - np.min(points[:, 1])
    padding: float = 0.1 * max(width, height)
    ax.set_xlim(np.min(points[:, 0]) - padding, np.max(points[:, 0]) + padding)
    ax.set_ylim(np.min(points[:, 1]) - padding, np.max(points[:, 1]) + padding)

    return fig, ax


def plot_loads(
    fig: Figure,
    ax: Axes,
    points: NDArray,
    loads: tuple[list[int], list[list[float]]],
    scale: float = 1.0,
    load_multiplier: float = 1.0,
    color: str = "red",
    text_fontsize: float = 8.0,
):
    """
    在节点上绘制载荷箭头。

    Args:
        fig (Figure): Matplotlib Figure 对象。
        ax (Axes): Matplotlib Axes 对象。
        points (NDArray): 节点坐标数组，形状为 (n, 2)。
        loads (tuple[list[int], list[list[float]]]): 载荷元组。
            第一项是节点索引列表。
            第二项是对应节点的载荷 [x, y] 分量列表。
        scale (float): 箭头整体缩放因子。
        load_multiplier (float): 载荷绘制长度的乘数。
        color (str): 箭头颜色。
        text_fontsize (float): 载荷文字标注大小。
    """
    node_indices, load_values = loads
    for node_idx, load_vec in zip(node_indices, load_values):
        if node_idx < len(points):
            start_point = points[node_idx]

            original_load_x = load_vec[0]
            original_load_y = load_vec[1]

            effective_dx_unscaled = original_load_x * load_multiplier
            effective_dy_unscaled = original_load_y * load_multiplier

            dx_arrow = effective_dx_unscaled * scale
            dy_arrow = effective_dy_unscaled * scale

            length_arrow = np.sqrt(dx_arrow**2 + dy_arrow**2)

            if length_arrow < 1e-9:
                text_content = f"({original_load_x:.2f}, {original_load_y:.2f})"
                ax.text(
                    start_point[0],
                    start_point[1] + 0.02 * scale,
                    text_content,
                    fontsize=text_fontsize,
                    color=color,
                    ha="center",
                    va="bottom",
                )
                continue

            arrow_tail_x = start_point[0] - dx_arrow
            arrow_tail_y = start_point[1] - dy_arrow

            current_head_width = 0.05 * scale
            current_head_length = 0.1 * scale

            if length_arrow < current_head_length * 1.5:
                current_head_length = length_arrow / 2.0
                current_head_width = current_head_length * 0.5

            current_head_width = max(current_head_width, 1e-6)
            current_head_length = max(current_head_length, 1e-6)

            ax.arrow(
                arrow_tail_x,
                arrow_tail_y,
                dx_arrow,
                dy_arrow,
                head_width=current_head_width,
                head_length=current_head_length,
                fc=color,
                ec=color,
                length_includes_head=True,
            )

            text_content = f"({original_load_x:.2f}, {original_load_y:.2f})"

            offset_val = current_head_length * 0.4

            text_x = start_point[0] + offset_val
            text_y = start_point[1] + offset_val
            ha_val = "left"
            va_val = "bottom"

            if dx_arrow < 0 and abs(dx_arrow) > abs(dy_arrow) * 0.7:
                text_x = start_point[0] - offset_val
                ha_val = "right"

            if dy_arrow < 0 and abs(dy_arrow) > abs(dx_arrow) * 0.7:
                text_y = start_point[1] - offset_val
                va_val = "top"

            if dy_arrow > 0 and abs(dy_arrow) > abs(dx_arrow) * 0.7 and dx_arrow >= 0:
                text_y = start_point[1] + offset_val
                va_val = "bottom"
            if dx_arrow > 0 and abs(dx_arrow) > abs(dy_arrow) * 0.7 and dy_arrow <= 0:
                text_x = start_point[0] + offset_val
                ha_val = "left"

            ax.text(
                text_x,
                text_y,
                text_content,
                fontsize=text_fontsize,
                color=color,
                ha=ha_val,
                va=va_val,
            )

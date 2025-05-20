import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.colors import Normalize

from numpy.typing import NDArray
import numpy as np


def plot_tri(
    triangles: NDArray,
    points: NDArray,
    stresses: NDArray | None = None,
    color: str = "b",
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
    assert triangles.shape[1] == 3, "三角形顶点编号数组的形状应为 (n, 3)"
    assert points.shape[1] == 2, "三角形顶点坐标数组的形状应为 (m, 2)"
    if stresses is not None:
        assert len(stresses) == len(triangles), (
            "应力值数组的长度应与三角形数组的长度一致"
        )

    fig, ax = plt.subplots()

    if stresses is not None:
        # 使用colormap绘制应力云图
        norm = Normalize(vmin=np.min(stresses), vmax=np.max(stresses))
        # 使用rainbow colormap表征应力状况
        cmap = cm.get_cmap("rainbow")

        for tri, stress in zip(triangles, stresses):
            pts = points[tri]
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
            pts = points[tri]
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

    return fig, ax

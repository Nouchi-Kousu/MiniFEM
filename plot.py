from matplotlib import lines
from mplfonts import use_font
import scienceplots
from matplotlib import pyplot as plt
import math
from matplotlib.patches import Circle
import numpy as np
import random
import procedures.mesh_generation as pr_mesh
from utils.plot import plot_tri
import utils.mesh as mesh


use_font("Noto Serif CJK SC")
plt.style.use(["science", "ieee", "no-latex", "cjk-sc-font"])


def circumscribed_circle_of_triangle(x1, y1, x2, y2, x3, y3):
    x0 = (
        (y2 - y1) * (y3 * y3 - y1 * y1 + x3 * x3 - x1 * x1)
        - (y3 - y1) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1)
    ) / (2 * (x3 - x1) * (y2 - y1) - 2 * (x2 - x1) * (y3 - y1))

    y0 = (
        (x2 - x1) * (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1)
        - (x3 - x1) * (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1)
    ) / (2 * (y3 - y1) * (x2 - x1) - 2 * (y2 - y1) * (x3 - x1))

    r = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return x0, y0, r


def main():
    points = np.array([[random.random() * 10, random.random() * 10] for _ in range(20)])
    point = np.array([5, 5])

    pt, tri = pr_mesh.gen_mesh_from_pts(points)

    fig, ax = plot_tri(
        pt,
        tri,
        color="#66ccff",
        linewidth=0.5,
    )
    ax.scatter(points[:, 0], points[:, 1], s=5, c="k", marker="o", label="原有结点", zorder=100)
    ax.scatter(point[0], point[1], s=5, c="r", marker="o", label="新结点", zorder=100)
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    ax.legend()
    plt.savefig("star-old.png", dpi=600, bbox_inches="tight")
    plt.close()

    del_bool = mesh.pt_in_circle(point, tri, pt)
    # 只保留有效三角形网格
    del_tri = tri[del_bool]

    # 构建星形多边形
    tris_unique_edges = mesh.tris_unique_edges(del_tri)

    fig, ax = plot_tri(
        pt,
        tri[~del_bool],
        color="#66ccff",
        linewidth=0.5,
    )

    for idx, edge in enumerate(tris_unique_edges):
        x1, y1 = pt[edge[0]]
        x2, y2 = pt[edge[1]]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="red",
            linewidth=1,
            linestyle="-",
            label="星形多边形" if idx == 0 else "",
        )

    ax.scatter(points[:, 0], points[:, 1], s=5, c="k", marker="o", zorder=100)
    ax.scatter(point[0], point[1], s=5, c="r", marker="o", zorder=100)
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    ax.legend()
    plt.savefig("star.png", dpi=600, bbox_inches="tight")
    plt.close()

    new_points = np.concatenate((pt, point.reshape(1, -1)), axis=0)
    _, new_tri = pr_mesh.gen_mesh_from_pts(new_points)

    fig, ax = plot_tri(
        new_points,
        new_tri,
        color="#66ccff",
        linewidth=0.5,
    )
    for idx, edge in enumerate(tris_unique_edges):
        x1, y1 = pt[edge[0]]
        x2, y2 = pt[edge[1]]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="red",
            linewidth=1,
            linestyle="-",
        )
        ax.plot(
            [point[0], x2],
            [point[1], y2],
            color="blue",
            linewidth=1,
            linestyle="-",
        )
        ax.plot(
            [point[0], x1],
            [point[1], y1],
            color="blue",
            linewidth=1,
            linestyle="-",
            label="新网格边" if idx == 0 else "",
        )

    ax.scatter(points[:, 0], points[:, 1], s=5, c="k", marker="o", zorder=100)
    ax.scatter(point[0], point[1], s=5, c="r", marker="o", zorder=100)
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    ax.legend()
    plt.savefig("star-new.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()

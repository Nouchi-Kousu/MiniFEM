from pyexpat import model
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import procedures.mesh_generation as pr_mesh
import utils.load as utils_load
from utils.plot import plot_tri, plot_loads
import utils.solve as utils_solve
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize
import scienceplots
from math import sqrt
from rich.console import Console
from mplfonts import use_font
from matplotlib.patches import Polygon
import pandas as pd

console = Console()


use_font("Noto Serif CJK SC")


def main():
    plt.style.use(["science", "ieee", "no-latex", "cjk-sc-font"])
    load_num = 3
    img_name = f"load{load_num}.png"

    model_point = np.array([[0, 0], [1, 0], [1, 5], [1, 7], [1, 10], [0, 10]])
    model_edge = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])

    # model_point = np.array(
    #     [
    #         [0, 0],
    #         [2, 0],
    #         [3, sqrt(3)],
    #         [2, 2 * sqrt(3)],
    #         [0, 2 * sqrt(3)],
    #         [-1, sqrt(3)],
    #         [1, sqrt(3)],
    #     ]
    # )
    # model_edge = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    fig, ax = plt.subplots()
    ax.add_patch(
        Polygon(
            model_point[model_edge].reshape(-1, 2),
            edgecolor="k",
            fill=True,
            facecolor="#66ccff",
            linewidth=0.4,
        )
    )
    ax.scatter(model_point[:, 0], model_point[:, 1], s=5, c="k", marker="o")
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    width: float = 1
    height: float = 10
    padding: float = 0.1 * max(width, height)
    ax.set_xlim(0 - padding, 1 + padding)
    ax.set_ylim(0 - padding, 10 + padding)
    ax.set_xticks([0, 1])
    plt.savefig("model.png", dpi=600, bbox_inches="tight")
    plt.close()

    mesh_size = 0.05
    points, index_list = pr_mesh.sample_pts_on_edges(
        model_point,
        model_edge,
        mesh_size,
    )
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=1, c="k", marker="o", label="布种点")
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    width: float = 4
    height: float = 2 * sqrt(3)
    padding: float = 0.1 * max(width, height)
    ax.set_xlim(-1 - padding, 3 + padding)
    ax.set_ylim(0 - padding, 2 * sqrt(3) + padding)
    plt.savefig("seed.png", dpi=600, bbox_inches="tight")
    plt.close()

    constraints = utils_load.collect_edge_dof_displacements_for_penalty(
        model_point,
        model_edge,
        index_list,
        [0, 4],
        [0.0, 0.0],
    )

    # load = utils_load.distribute_edge_loads(
    #     model_point,
    #     model_edge,
    #     index_list,
    #     3,
    #     [0.0, -10.0],
    # )

    load = ([3], [[-1.0, 0.0]])

    pt, tri = pr_mesh.gen_mesh_from_pts(points)
    # fig, ax = plot_tri(pt, tri, linewidth=0.4)
    # ax.set_aspect("equal")
    # ax.set_xlabel("X 坐标")
    # ax.set_ylabel("Y 坐标")
    # plt.savefig("fast_mesh.png", dpi=600, bbox_inches="tight")
    # plt.close()

    pt, tri = pr_mesh.refine_mesh(pt, tri, mesh_size)

    K_global = utils_solve.assemble_global_stiffness(pt, tri, E=100, nu=0.3, t=1)
    P_global = utils_solve.P_global(load, pt.shape[0] * 2)

    K_global, P_global = utils_solve.enforce_constraints_diagonal_fix(
        constraints,
        K_global,
        P_global,
    )

    u: NDArray = np.array(spsolve(K_global, P_global).reshape(-1, 2))

    pt += u  # 放大位移以便可视化

    S_list = utils_solve.S_list(pt, tri, E=1, nu=0.3)

    u_list = utils_solve.u_list(pt, tri, u)

    sigma = S_list @ u_list

    mises_sigma = np.sqrt(
        (sigma[:, 0] + sigma[:, 1]) ** 2
        - 3 * (sigma[:, 0] * sigma[:, 1] - sigma[:, 2] ** 2)
    )
    console.log(
        f"[blue]von Mises 应力范围: {np.min(mises_sigma)} - {np.max(mises_sigma)}[/blue]"
    )

    # fig, ax = plot_tri(pt, tri, stresses=mises_sigma, linewidth=0.1)
    fig, ax = plot_tri(pt, tri, linewidth=0.01, stresses=mises_sigma)
    plot_loads(fig, ax, pt, load, scale=1, load_multiplier=1, text_fontsize=4)
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    # ax.set_title("带应力的三角形网格")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="rainbow",
            norm=Normalize(vmin=np.min(mises_sigma), vmax=np.max(mises_sigma)),
        ),
        ax=ax,
        label="Mises 应力",
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks(np.linspace(np.min(mises_sigma), np.max(mises_sigma), 5))

    plt.savefig(img_name, dpi=600, bbox_inches="tight")
    console.log(f"[blue]网格绘制完成，保存为 {img_name}[/blue]")
    print(f"2点位移为: {u[2]}")
    print(f"3点位移为: {u[3]}")

    # 保存 pt, tri, u 到 CSV 文件
    pt_df = pd.DataFrame(pt, columns=["x", "y"])
    pt_df.to_csv(f"pt_{img_name}.csv", index=False)
    console.log(f"[blue]pt 数据已保存到 pt_{img_name}.csv[/blue]")

    tri_df = pd.DataFrame(tri, columns=["pt0", "pt1", "pt2"])
    tri_df.to_csv(f"tri_{img_name}.csv", index=False)
    console.log(f"[blue]tri 数据已保存到 tri_{img_name}.csv[/blue]")

    u_df = pd.DataFrame(u, columns=["ux", "uy"])
    u_df.to_csv(f"u_{img_name}.csv", index=False)
    console.log(f"[blue]u 数据已保存到 u_{img_name}.csv[/blue]")


if __name__ == "__main__":
    main()

import random
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from rich import print
from numpy.typing import NDArray
from utils.plot import plot_tri
from matplotlib.colors import Normalize
from mplfonts import use_font
from utils.utils import (
    tri_area,
    tri_centroid,
    pt2tri_max_dist,
)
from utils.mesh import pt_in_circle, tris_unique_edges
import procedures.mesh_generation as pr_mesh
import utils.load as utils_load

use_font("Noto Serif CJK SC")


def main():
    # points: NDArray = np.array(
    #     [
    #         [i + random.random() / 10, j + random.random() / 10]
    #         for i in range(-3, 4)
    #         for j in range(-3, 4)
    #     ],
    #     dtype=float,
    # )
    # # points = np.array([[-3, -3], [-3, 3], [0.1, 0.1]])
    # # tri = np.array([[0, 1, 2]])
    # # print(pt_in_circle(np.array([1, 0]), tri, points))
    # pt, tri = pr_mesh.gen_mesh_from_pts(points)

    # fig, ax = plot_tri(pt, tri)
    # plt.show()
    points, index_list = pr_mesh.sample_pts_on_edges(
        np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
        1.1,
    )
    # plt.scatter(points[:, 0], points[:, 1], s=1, c="k", marker="o")
    # plt.show()

    print(
        utils_load.apply_edge_constraints(
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]]),
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
            index_list,
            [1],
            [True, False],
        )
    )

    # plt.scatter(points[:, 0], points[:, 1], s=1, c="k", marker="o")
    # plt.show()

    # pt, tri = pr_mesh.gen_mesh_from_pts(points)

    # fig, ax = plot_tri(pt, tri)
    # plt.savefig("mesh-.png", dpi=300, bbox_inches="tight")

    # pt, tri = pr_mesh.refine_mesh(pt, tri, 0.1)

    # fig, ax = plot_tri(pt, tri)
    # plt.savefig("mesh.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

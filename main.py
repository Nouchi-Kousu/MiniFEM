import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import procedures.mesh_generation as pr_mesh
import utils.load as utils_load
from utils.plot import plot_tri
import utils.solve as utils_solve
from scipy.sparse.linalg import spsolve


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
    model_point = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    model_edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    points, index_list = pr_mesh.sample_pts_on_edges(
        model_point,
        model_edge,
        0.05,
    )
    # plt.scatter(points[:, 0], points[:, 1], s=1, c="k", marker="o")
    # plt.show()

    constraints = utils_load.collect_edge_dof_displacements_for_penalty(
        model_point,
        model_edge,
        index_list,
        [3],
        [0, 0],
    )

    load = utils_load.distribute_edge_loads(
        model_point,
        model_edge,
        index_list,
        [1],
        [-1, -1],
    )

    pt, tri = pr_mesh.gen_mesh_from_pts(points)

    pt, tri = pr_mesh.refine_mesh(pt, tri, 0.05)

    K_global = utils_solve.assemble_global_stiffness(pt, tri, E=1.0, nu=0.3, t=1.0)

    P_global = utils_solve.P_global(load)

    u = spsolve(K_global, P_global).reshape(-1, 2)

    pt -= u

    fig, ax = plot_tri(pt, tri)
    plt.savefig("mesh.png", dpi=300, bbox_inches="tight")


    # plt.scatter(points[:, 0], points[:, 1], s=1, c="k", marker="o")
    # plt.show()

    # b_list: tuple = utils_solve.b_list(points, tri)
    # c_list: tuple = utils_solve.c_list(points, tri)
    # K_rs: NDArray = utils_solve.get_K_rs(points, tri, b_list, c_list, 1, 0)
    # print("K_rs:", K_rs)
    # print(utils_solve.assemble_global_stiffness(pt, tri, E=1.0, nu=0.3, t=1.0))

    # fig, ax = plot_tri(pt, tri)
    # plt.savefig("mesh-.png", dpi=300, bbox_inches="tight")

    # pt, tri = pr_mesh.refine_mesh(pt, tri, 0.1)

    # fig, ax = plot_tri(pt, tri)
    # plt.savefig("mesh.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

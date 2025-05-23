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

use_font("Noto Serif CJK SC")


def main():
    # plt.style.use(["science", "nature", "no-latex", "cjk-sc-font"])
    triangles: NDArray = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    points: NDArray = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 1.5]], dtype=float
    )
    # stresses: NDArray = np.array([1, 2, 3], dtype=float)

    # fig, ax = plot_tri(triangles, points, stresses=stresses)
    # x_min: float = np.min(points[:, 0])
    # x_max: float = np.max(points[:, 0])
    # x_len: float = x_max - x_min
    # y_min: float = np.min(points[:, 1])
    # y_max: float = np.max(points[:, 1])
    # y_len: float = y_max - y_min
    # padding: float = 0.1 * max(x_len, y_len)
    # ax.set_xlim(x_min - padding, x_max + padding)
    # ax.set_ylim(y_min - padding, y_max + padding)
    # ax.set_aspect("equal")
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_title("带应力的三角形网格")
    # fig.colorbar(
    #     plt.cm.ScalarMappable(
    #         cmap="rainbow", norm=Normalize(vmin=np.min(stresses), vmax=np.max(stresses))
    #     ),
    #     ax=ax,
    #     label="Stress",
    # )
    # plt.savefig("triangle_mesh.png", dpi=600, bbox_inches="tight")

    print(tris_unique_edges(triangles))


if __name__ == "__main__":
    main()

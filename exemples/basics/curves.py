from os.path import join, dirname

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

from shaft import curve_transform, triangulate_curve, mesh_curve


def main():
    leaf_data = join(dirname(__file__), "../../data/curves/leaves", "curves.npy")
    curves = np.load(leaf_data)
    curve = curves[0] / 1.5
    triangles = triangulate_curve(curve)

    k_range = 100
    ks = np.mgrid[1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1,].reshape(2, -1).T
    fcs = curve_transform(triangles, ks)

    ###### Now some plots ######

    # To plot meshed curve:
    # curve_mesh = mesh_curve(curve)
    # curve_mesh.plot(show_edges=True)

    plt.title(f"The first {len(fcs)} features for 2 different curves")
    plt.plot(fcs[:, 0], label="Sin")
    plt.plot(fcs[:, 1], label="Cos")
    plt.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Coefs wrt k_range of first curve", fontsize=20)
    ax1.set_title("Sin coefs")
    ax1.imshow(fcs[:, 0].reshape(k_range,k_range))
    ax2.set_title("Cos coefs")
    ax2.imshow(fcs[:, 1].reshape(k_range,k_range))
    plt.show()

    breakpoint()


if __name__ == "__main__":
    main()

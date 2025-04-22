from os.path import join, dirname

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


def triangulate_curve(curve):
    zer = np.zeros((len(curve), 1))
    curve_points = np.append(curve, zer, axis=1)
    l1 = np.arange(0, len(curve_points))
    lines = np.concatenate((np.array([len(curve_points)]), l1))
    curve = pv.PolyData(curve_points, lines)
    mesh = curve.triangulate()
    faces = mesh.regular_faces
    points = mesh.points
    points = points[:, :2]
    triangles = points[faces]
    return triangles


def fourier_features(triangles, k_range):
    ks = np.mgrid[0 : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1].reshape(2, -1).T
    # Get M and det(M)
    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)

    # Get the (M.t*k)
    mtks = np.matvec(mt, ks[:, np.newaxis, :]).transpose(
        1, 0, 2
    )  # TODO: Check if this is correct

    # Get the base fourier coeffs for the other triangles
    a = np.power(mtks[:, :, 1] * (mtks[:, :, 1] - mtks[:, :, 0]), -1)
    b = np.power(mtks[:, :, 0] * mtks[:, :, 1], -1)
    sTks = a * np.sin(mtks[:, :, 0]) + (b - a) * np.sin(mtks[:, :, 1])
    cTks = a * np.cos(mtks[:, :, 1]) + (a + b) * np.cos(mtks[:, :, 0]) - b
    # TODO: Figure out what to do with the NANS

    # Get the fourier coeffs for each triangles
    det_m = np.linalg.det(m)
    cos = np.cos(np.inner(ks, triangles[:, 2])).T
    sin = np.sin(np.inner(ks, triangles[:, 2])).T
    stk = det_m[:, np.newaxis] * (sTks * cos + cTks * sin)
    ctk = det_m[:, np.newaxis] * (cTks * cos - sTks * sin)
    # TODO: check if inner is the correct function to use here

    # Sum and stack to get the shape fourier features
    triangles_features = np.stack((stk, ctk), axis=-1)
    shape_features = np.sum(triangles_features, axis=0)
    # Get rid of the nans here:
    shape_features = shape_features[~np.isnan(shape_features)].reshape((-1, 2))

    return shape_features


def main():
    leaf_data = join(dirname(__file__), "../../data/curves/leaves", "curves.npy")
    curves = np.load(leaf_data)
    curve_0 = curves[0] / 1.5
    curve_1 = curves[1] / 1.5

    k_range = 200

    triangles = triangulate_curve(curve_0)
    fourier_feats_0 = fourier_features(triangles, k_range)

    triangles = triangulate_curve(curve_1)
    fourier_feats_1 = fourier_features(triangles, k_range)

    plt.title(f"The first {len(fourier_feats_0)} fourier features")
    plt.plot(fourier_feats_0[:, 0], label="S_k1")
    plt.plot(fourier_feats_0[:, 1], label="C_k1")
    plt.plot(fourier_feats_1[:, 0], label="S_k2")
    plt.plot(fourier_feats_1[:, 1], label="C_k2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

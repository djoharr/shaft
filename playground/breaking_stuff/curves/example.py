from os.path import join, dirname

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


def mesh_curve(curve):
    zer = np.zeros((len(curve), 1))
    curve_points = np.append(curve, zer, axis=1)
    l1 = np.arange(0, len(curve_points))
    lines = np.concatenate((np.array([len(curve_points)]), l1))
    curve = pv.PolyData(curve_points, lines)
    mesh = curve.triangulate()
    return mesh


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

    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(m)

    mtks = np.einsum('ijk,lj->lik', mt, ks).transpose(1, 0, 2)
    a = np.power(mtks[:, :, 1] * (mtks[:, :, 0] - mtks[:, :, 1]), -1)
    b = np.power(mtks[:, :, 0] * mtks[:, :, 1], -1)
    sin_k1, sin_k2 = np.sin(mtks[:, :, 0]), np.sin(mtks[:, :, 1])
    cos_k1, cos_k2 = np.cos(mtks[:, :, 0]), np.cos(mtks[:, :, 1])
    sTks = a * (sin_k2 - sin_k1) + b * sin_k1
    cTks = a * (cos_k2 - cos_k1) + b * (cos_k1 - 1)

    cos = np.cos(np.inner(ks, triangles[:, 2])).T
    sin = np.sin(np.inner(ks, triangles[:, 2])).T
    stk = det_m[:, np.newaxis] * (sTks * cos + cTks * sin)
    ctk = det_m[:, np.newaxis] * (cTks * cos - sTks * sin)

    triangles_features = np.stack((stk, ctk), axis=-1)
    shape_features = np.sum(triangles_features, axis=0)

    return shape_features


def main():
    leaf_data = join(dirname(__file__), "../../data/curves/leaves", "curves.npy")
    curves = np.load(leaf_data)
    curve_0 = curves[0] / 1.5
    curve_1 = curves[1] / 1.5

    k_range = 100

    triangles = triangulate_curve(curve_0)
    fourier_feats_0 = fourier_features(triangles, k_range)

    triangles = triangulate_curve(curve_1)
    fourier_feats_1 = fourier_features(triangles, k_range)

    curve_1_mesh = mesh_curve(curve_0)
    curve_1_mesh.plot(show_edges=True)

    plt.title(f"The first {len(fourier_feats_0)} features for 2 different curves")
    plt.plot(fourier_feats_0[:, 0], label="S_k1")
    plt.plot(fourier_feats_0[:, 1], label="C_k1")
    plt.plot(fourier_feats_1[:, 0], label="S_k2")
    plt.plot(fourier_feats_1[:, 1], label="C_k2")
    plt.legend()
    plt.show()

    Sks = fourier_feats_0[:, 0]
    Cks = fourier_feats_0[:, 1]
    Sks_xy = Sks[~np.isnan(Sks)].reshape(202, 100)
    Cks_xy = Cks[~np.isnan(Cks)].reshape(202, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Coefs wrt k_range of first curve", fontsize=20)
    ax1.set_title("Sks coefs")
    ax1.imshow(Sks_xy)
    ax2.set_title("Cks coefs")
    ax2.imshow(Cks_xy)
    plt.show()

    breakpoint()


if __name__ == "__main__":
    main()

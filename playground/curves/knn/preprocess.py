from os.path import join, dirname

import pyvista as pv
import numpy as np
from tqdm import tqdm

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

def loopy_fourier(triangles, k_range):
    ks = np.mgrid[0 : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1].reshape(2, -1).T

    Sks = []
    Cks = []
    for triangle in triangles:
        M = np.array([triangle[0] - triangle[2], triangle[1] - triangle[2]])
        Mt = M.T
        detM = np.linalg.det(Mt)

        Mtk = np.matvec(Mt, ks)
        k1, k2 = Mtk.T
        coef_1 = [(k2[i] * (k1[i] - k2[i])) ** (-1) for i in range(len(k1))]
        coef_2 = [(k1[i] * k2[i]) ** (-1) for i in range(len(k1))]
        Stmtk = coef_1 * (np.sin(k2) - np.sin(k1)) + coef_2 * np.sin(k1)
        Ctmtk = coef_1 * (np.cos(k2) - np.cos(k1)) + coef_2 * (np.cos(k1) - 1)

        Sk_1 = detM * np.cos(np.dot(ks, triangle[2])) * Stmtk
        Sk_2 = detM * np.sin(np.dot(ks, triangle[2])) * Ctmtk
        Ck_1 = detM * np.cos(np.dot(ks, triangle[2])) * Ctmtk
        Ck_2 = detM * np.sin(np.dot(ks, triangle[2])) * Stmtk
        Sk = Sk_1 + Sk_2
        Ck = Ck_1 - Ck_2

        Sks.append(Sk)
        Cks.append(Ck)

    fourier_sin = np.sum(np.array(Sks), axis=0)
    fourier_cos = np.sum(np.array(Cks), axis=0)
    shape_features = np.stack((fourier_sin, fourier_cos), axis=-1)

    return shape_features

def fourier_features(triangles, k_range, drop_nans=True):
    ks = np.mgrid[0 : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1].reshape(2, -1).T

    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(m)

    mtks = np.matvec(mt, ks[:, np.newaxis, :]).transpose(1, 0, 2)
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

    if drop_nans:
        shape_features = shape_features[~np.isnan(shape_features)].reshape((-1, 2))

    return shape_features


def main():
    leaf_data = join(dirname(__file__), "../../../data/curves/leaves", "curves.npy")
    all_curves = np.load(leaf_data)

    all_coefs = []
    for curve in tqdm(all_curves):
        curve = curve / 1.5

        k_range = 12
        triangles = triangulate_curve(curve)
        # fourier_coefs = fourier_features(triangles, k_range, drop_nans=False)
        fourier_coefs = loopy_fourier(triangles, k_range)
        all_coefs.append(fourier_coefs)

    np.save(join(dirname(__file__), "../../../data/curves/leaves/fourier_coefs_12.npy"), all_coefs)

    breakpoint()


if __name__ == "__main__":
    main()

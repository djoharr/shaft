from os.path import join, dirname

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import time


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

def get_idxs(ks):
    all_idx = list(range(0,len(ks)))

    null_x_idx = np.argwhere(ks[:, 0]==0).flatten()
    null_y_idx = np.argwhere(ks[:, 1]==0).flatten()
    equ_xy_idx = np.argwhere(ks[:, 1]==ks[:, 0]).flatten()

    null_xy_idx = list(set(null_x_idx) & set(null_y_idx))
    if null_xy_idx:
        null_x_idx = np.delete(null_x_idx, np.argwhere(null_x_idx==null_xy_idx))
        null_y_idx = np.delete(null_y_idx, np.argwhere(null_y_idx==null_xy_idx))
        equ_xy_idx = np.delete(equ_xy_idx, np.argwhere(equ_xy_idx==null_xy_idx))

    rest_idx = list(set(all_idx) - set(null_xy_idx) - set(null_x_idx) - set(null_y_idx) - set(equ_xy_idx))

    return rest_idx, null_xy_idx, null_x_idx, null_y_idx, equ_xy_idx

def get_sc(ks, mat):
    mat_base = np.einsum('ijk,lj->lik', mat, ks).transpose(1, 0, 2)
    sinx, siny = np.sin(mat_base[:, :, 0]), np.sin(mat_base[:, :, 1])
    cosx, cosy = np.cos(mat_base[:, :, 0]), np.cos(mat_base[:, :, 1])

    normies, nxy, nx, ny, exy = get_idxs(ks)
    sins = np.zeros(mat_base.shape[:2])
    coss = np.zeros(mat_base.shape[:2])

    mnorm = mat_base[:, normies, :]
    sinx_, siny_ = sinx[:, normies], siny[:, normies]
    cosx_, cosy_ = cosx[:, normies], cosy[:, normies]
    a = np.power(mnorm[:, :, 1] * (mnorm[:, :, 0] - mnorm[:, :, 1]), -1)
    b = np.power(mnorm[:, :, 0] * mnorm[:, :, 1], -1)

    sins[:, normies] = a * (siny_ - sinx_) + b * sinx_
    coss[:, normies] = a * (cosy_ - cosx_) + b * (cosx_ - 1)

    mnxy = mat_base[:, nxy, :]
    sins[:, nxy] = 0
    coss[:, nxy] = 0

    mnx = mat_base[:, nx, :]
    siny_, cosy_ = siny[:, nx], cosy[:, nx]
    sins[:, nx] = - np.power(mnx[:, :, 1], -2) * siny_ + np.power(mnx[:, :, 1], -1)
    coss[:, nx] = - np.power(mnx[:, :, 1], -2) * (cosy_ + 1)

    mny = mat_base[:, ny, :]
    sinx_, cosx_ = sinx[:, ny], cosx[:, ny]
    sins[:, ny] = - np.power(mny[:, :, 0], -2) * sinx_ + np.power(mny[:, :, 0], -1)
    coss[:, ny] = np.power(mny[:, :, 0], -2) * (1 - cosx_)

    mexy = mat_base[:, exy, :]
    sinx_, cosx_ = sinx[:, exy], cosx[:, exy]
    sins[:, exy] = - np.power(mexy[:, :, 0], -1) * cosx_ + np.power(mexy[:, :, 0], -2) * sinx_
    coss[:, exy] = np.power(mexy[:, :, 0], -2) * (cosx_ - 1) + np.power(mexy[:, :, 0], -1) * sinx_

    return sins, coss



def fourier_features(triangles, k_range, compute_taylor=False):
    ks = np.mgrid[1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1,].reshape(2, -1).T

    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(m)

    # mat = np.matvec(mt, ks[:, np.newaxis, :]).transpose(1, 0, 2)
    if compute_taylor:
        stk, ctk = get_sc(ks, mt)
    else:
        mat = np.einsum('ijk,lj->lik', mt, ks).transpose(1, 0, 2)
        a = np.power(mat[:, :, 1] * (mat[:, :, 0] - mat[:, :, 1]), -1)
        b = np.power(mat[:, :, 0] * mat[:, :, 1], -1)
        sinx, siny = np.sin(mat[:, :, 0]), np.sin(mat[:, :, 1])
        cosx, cosy = np.cos(mat[:, :, 0]), np.cos(mat[:, :, 1])
        stk = a * (siny - sinx) + b * sinx
        ctk = a * (cosy - cosx) + b * (cosx - 1)


    cos = np.cos(np.inner(ks, triangles[:, 2])).T
    sin = np.sin(np.inner(ks, triangles[:, 2])).T
    sin_coefs = det_m[:, np.newaxis] * (stk * cos + ctk * sin)
    cos_coefs = det_m[:, np.newaxis] * (ctk * cos - stk * sin)

    triangles_features = np.stack((sin_coefs, cos_coefs), axis=-1)
    shape_features = - np.sum(triangles_features, axis=0)

    return shape_features



def main():
    leaf_data = join(dirname(__file__), "../../../data/curves/leaves", "curves.npy")
    curves = np.load(leaf_data)
    curve = curves[0] / 1.5

    triangles = triangulate_curve(curve)

    k_range = 100

    t0 = time.time()

    fcs = fourier_features(triangles, k_range)

    t1 = time.time()

    fcs_taylor = fourier_features(triangles, k_range, compute_taylor=True)

    t2 = time.time()

    print(f"Number of coeffs: {len(fcs)*2}")
    print(f"Time taken for normal: {t1 - t0} seconds")
    print(f"Time taken for taylor: {t2 - t1} seconds")

    # plt.title(f"The first {len(fcs)} features for a sample curve")
    # plt.plot(fcs[:, 0], label="Sin")
    # plt.plot(fcs[:, 1], label="Cos")
    # plt.show()

    breakpoint()


if __name__ == "__main__":
    main()

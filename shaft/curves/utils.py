import numpy as np
import pyvista as pv


def mesh_curve(curve):
    zer = np.zeros((len(curve), 1))
    curve_points = np.append(curve, zer, axis=1)
    l1 = np.arange(0, len(curve_points))
    lines = np.concatenate((np.array([len(curve_points)]), l1))
    curve = pv.PolyData(curve_points, lines)
    mesh = curve.triangulate()
    return mesh


def triangulate_curve(curve):
    mesh = mesh_curve(curve)
    faces = mesh.regular_faces
    points = mesh.points
    points = points[:, :2]
    triangles = points[faces]
    return triangles



def get_masks(ks):
    null_x, null_y, eq_xy = ks[:, :, 0] == 0, ks[:, :, 1] == 0, ks[:,:,0]==ks[:,:,1]
    null_xy = null_x & null_y
    null_x = null_x & ~null_xy
    null_y = null_y & ~null_xy
    eq_xy = eq_xy & ~null_xy
    rest = ~null_x & ~null_y & ~null_xy & ~eq_xy
    return rest, null_xy, null_x, null_y, eq_xy

def get_sc(ks, mat):
    mat_base = np.einsum('ijk,lj->lik', mat, ks)
    sinx, siny = np.sin(mat_base[:, :, 0]), np.sin(mat_base[:, :, 1])
    cosx, cosy = np.cos(mat_base[:, :, 0]), np.cos(mat_base[:, :, 1])
    normies, nxy, nx, ny, exy = get_masks(mat_base)

    sins = np.zeros(mat_base.shape[:2])
    coss = np.zeros(mat_base.shape[:2])

    mnorm = mat_base[normies]
    sinx_, siny_ = sinx[normies], siny[normies]
    cosx_, cosy_ = cosx[normies], cosy[normies]

    a = np.power(mnorm[:, 1] * (mnorm[:, 0] - mnorm[:, 1]), -1)
    b = np.power(mnorm[:, 0] * mnorm[:, 1], -1)
    sins[normies] =  a * (siny_ - sinx_) + b * sinx_
    coss[normies] = a * (cosy_ - cosx_) + b * (cosx_ - 1)

    mnxy = mat_base[nxy]
    sins[nxy] = 0
    coss[nxy] = 0

    mnx = mat_base[nx]
    siny_, cosy_ = siny[nx], cosy[nx]
    sins[nx] = - np.power(mnx[:, 1], -2) * siny_ + np.power(mnx[:, 1], -1)
    coss[nx] = - np.power(mnx[:, 1], -2) * (cosy_ + 1)

    mny = mat_base[ny]
    sinx_, cosx_ = sinx[ny], cosx[ny]
    sins[ny] = - np.power(mny[:, 0], -2) * sinx_ + np.power(mny[:, 0], -1)
    coss[ny] = np.power(mny[:, 0], -2) * (1 - cosx_)

    mexy = mat_base[exy]
    sinx_, cosx_ = sinx[exy], cosx[exy]
    sins[exy] = - np.power(mexy[:, 0], -1) * cosx_ + np.power(mexy[:, 0], -2) * sinx_
    coss[exy] = np.power(mexy[:, 0], -2) * (cosx_ - 1) + np.power(mexy[:, 0], -1) * sinx_

    return sins.T, coss.T

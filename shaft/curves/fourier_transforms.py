import numpy as np
from .utils import triangulate_curve, get_sc


def curve_transform(triangles, lattice):
    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(m)

    stk, ctk = get_sc(lattice, mt)
    cos = np.cos(np.inner(lattice, triangles[:, 2])).T
    sin = np.sin(np.inner(lattice, triangles[:, 2])).T
    sin_coefs = det_m[:, np.newaxis] * (stk * cos + ctk * sin)
    cos_coefs = det_m[:, np.newaxis] * (ctk * cos - stk * sin)
    triangles_features = np.stack((sin_coefs, cos_coefs), axis=-1)
    shape_features = - np.sum(triangles_features, axis=0)

    return shape_features


def cft(curve, lattice_range=50):
    lattice = np.mgrid[0 : lattice_range - .99 : 1, 0 : lattice_range -.99 : 1,].reshape(2, -1).T
    triangles = triangulate_curve(curve)
    shape_features = curve_transform(triangles, lattice)

    return shape_features

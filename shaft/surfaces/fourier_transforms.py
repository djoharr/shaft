import numpy as np
from .taylor_monsters import get_sc, get_sc_no_taylor
from .utils import tetrahedralize


def surface_transform(tetras, lattice, taylor=True):
    m = tetras[:, :3] - tetras[:, 3][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(mt)

    if taylor:
        stk, ctk = get_sc(lattice, mt)
    else:
        stk, ctk = get_sc_no_taylor(lattice, mt)

    cos = np.cos(np.inner(lattice, tetras[:, 3])).T
    sin = np.sin(np.inner(lattice, tetras[:, 3])).T
    sin_coefs = det_m[:, np.newaxis] * (stk * cos + ctk * sin)
    cos_coefs = det_m[:, np.newaxis] * (ctk * cos - stk * sin)
    tetras_features = np.stack((sin_coefs, cos_coefs), axis=-1)
    shape_features = - np.sum(tetras_features, axis=0)

    return shape_features


def sft(surface, k_range=50, taylor=True):
    lattice = np.mgrid[0 : k_range - .99 : 1, 0 : k_range - .99 : 1, 0 : k_range - .99 : 1,].reshape(3, -1).T
    tetras = tetrahedralize(surface)
    coefs = surface_transform(tetras, lattice, taylor)
    return coefs

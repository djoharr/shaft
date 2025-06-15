from os.path import join, dirname
import numpy as np
import time
import pyvista as pv
import tetgen

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def tetrahedralize(surface):
    tet = tetgen.TetGen(surface)
    points, tets = tet.tetrahedralize()
    tetras = points[tets]
    return tetras


def fourier_features(tetras, ks):

    M = tetras[:, :3] - tetras[:, 3][:, np.newaxis]
    M = M.transpose(0, 2, 1)
    det_M = np.linalg.det(M)

    mtks = np.matvec(M, ks[:, np.newaxis, :]).transpose(1, 0, 2)
    # mtks = np.einsum('ijk,lj->lik', M, ks).transpose(1, 0, 2)

    x, y, z = mtks[:, :, 0], mtks[:, :, 1], mtks[:, :, 2]
    a = np.power(z * (y-z) * (x-y), -1)
    b = np.power(z * (y-z) * (x-z), -1)
    c = np.power(y * z * (x-y), -1)
    d = np.power(x * y * z, -1)
    sinx, siny, sinz = np.sin(x), np.sin(y), np.sin(z)
    cosx, cosy, cosz = np.cos(x), np.cos(y), np.cos(z)

    cTks = - a * (sinx - siny) + b * (sinx - sinz) + c * (sinx - siny) - d * sinx
    sTks = a * (cosx - cosy) - b * (cosx - cosz) - c * (cosx - cosy) + d * (cosx - 1)

    cos = np.cos(np.inner(ks, tetras[:, 3])).T
    sin = np.sin(np.inner(ks, tetras[:, 3])).T

    stk = det_M[:, np.newaxis] * (sTks * cos + cTks * sin)
    ctk = det_M[:, np.newaxis] * (cTks * cos - sTks * sin)

    tetras_features = np.stack((stk, ctk), axis=-1)
    shape_features = - np.sum(tetras_features, axis=0)

    return shape_features


def main():
    mesh_path = join(dirname(__file__), "../../data/surfaces/shrec", "100.vtk")
    mesh = pv.read(mesh_path)


    tetras = tetrahedralize(mesh)
    k_range = 10
    ks = np.mgrid[1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1,].reshape(3, -1).T
    fcs = fourier_features(tetras, ks)

    abs_coefs = np.sqrt(fcs[:,0]**2 + fcs[:,1]**2)

    ### Some plots
    x, y, z = ks[:,0], ks[:,1], ks[:,2]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Sin", "Cos"),
                        specs=[[{'type': 'volume'}, {'type': 'volume'}, {'type': 'volume'}]])

    fig.add_trace(go.Volume(x=x, y=y, z=z, value=fcs[:,0], isomin=fcs.min(), isomax=fcs.max(),
                            opacity=0.1, surface_count=30, colorscale='viridis'), row=1, col=1)
    fig.add_trace(go.Volume(x=x, y=y, z=z, value=fcs[:,1], isomin=fcs.min(), isomax=fcs.max(),
                            opacity=0.1, surface_count=30, colorscale='viridis'), row=1, col=2)
    fig.add_trace(go.Volume(x=x, y=y, z=z, value=abs_coefs, isomin=abs_coefs.min(), isomax=abs_coefs.max(),
                            opacity=0.1, surface_count=30, colorscale='viridis'), row=1, col=3)

    fig.update_layout(height=600, width=1200, title_text="Full coefficients")
    fig.show()


    plt.title(f"The first {len(fcs)} features for a sample shape")
    plt.plot(fcs[:, 0], label="Sin")
    plt.plot(fcs[:, 1], label="Cos")
    plt.plot(abs_coefs, label="Norm")
    plt.legend()
    plt.show()

    breakpoint()


if __name__ == "__main__":
    main()

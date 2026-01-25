from os.path import join, dirname
import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shaft import tetrahedralize, surface_transform


def main():
    surface_path = join(dirname(__file__), "../../data/surfaces/shrec", "0.vtk")
    surface = pv.read(surface_path)

    tetras = tetrahedralize(surface)
    k_range = 10
    ks = np.mgrid[1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1,].reshape(3, -1).T
    fcs = surface_transform(tetras, ks)

    ###### Now some plots ######
    # To plot the surface:
    # surface.plot()

    x, y, z = ks[:,0], ks[:,1], ks[:,2]
    abs_coefs = np.sqrt(fcs[:,0]**2 + fcs[:,1]**2)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Sin", "Cos", "Norm"),
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

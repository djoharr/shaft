from os.path import dirname, join

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pyvista as pv

from shaft import surface_transform, tetrahedralize


def main():
    surface_path = join(dirname(__file__), "../../data/assets/cheburashka.off")
    surface = pv.read(surface_path)
    surface = surface.extract_surface()
    tetras = tetrahedralize(surface)
    k_range = 10
    ks = (
        np.mgrid[
            1 : k_range + 0.1 : 1,
            1 : k_range + 0.1 : 1,
            1 : k_range + 0.1 : 1,
        ]
        .reshape(3, -1)
        .T
    )
    fcs = surface_transform(tetras, ks)

    surface.plot()

    coefs = fcs[:, 0]
    x, y, z = ks[:, 0], ks[:, 1], ks[:, 2]
    fig = go.Figure(
        data=go.Volume(
            x=x,
            y=y,
            z=z,
            value=coefs,
            isomin=coefs.min(),
            isomax=coefs.max(),
            opacity=0.2,
            surface_count=45,
            colorscale="viridis",
        )
    )

    # Helix equation
    t = np.linspace(0, 10, 50)
    x, y, z = np.cos(t), np.sin(t), t

    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5

    fig.update_layout(
        title="Rotating coefs",
        width=600,
        height=600,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1,
                x=0.8,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=5, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
    )

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    frames = []
    for t in np.arange(0, 18, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames = frames

    fig.show()

    pio.write_html(fig, file="animation.gif", auto_play=True)

    breakpoint()


if __name__ == "__main__":
    main()

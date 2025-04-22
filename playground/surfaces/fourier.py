from os.path import join, dirname

import pyvista as pv
import tetgen


def tetrahedralize(surface):
    tet = tetgen.TetGen(surface)
    points, tets = tet.tetrahedralize()
    tetras = points[tets]
    return tetras


def main():
    mesh_path = join(dirname(__file__), "../../data/surfaces/shrec", "0.vtk")
    mesh = pv.read(mesh_path)

    tetras = tetrahedralize(mesh)

    print(
        f"It starts with {mesh.n_points} points for the surface, and ends with {tetras.shape[0]} tetrahedra"
    )

    print(f"Fourier features TBD")


if __name__ == "__main__":
    main()

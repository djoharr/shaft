import tetgen
import pymeshfix as mf


def fix_mesh(mesh):
    meshfix = mf.MeshFix(mesh)
    meshfix.repair()
    return meshfix.mesh


def tetrahedralize(surface):
    mesh = fix_mesh(surface)
    tet = tetgen.TetGen(mesh)
    points, tets = tet.tetrahedralize()
    tetras = points[tets]
    return tetras

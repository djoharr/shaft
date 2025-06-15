from os.path import join, dirname
import numpy as np
import time
import pyvista as pv
import tetgen


def tetrahedralize(surface):
    tet = tetgen.TetGen(surface)
    points, tets = tet.tetrahedralize()
    tetras = points[tets]
    return tetras


def M_getter(Tet_list):
	Tet_mat_4=[]
	for i in range(len(Tet_list)):
		Tet_mat_4.append(np.array([[Tet_list[i][0][0]-Tet_list[i][3][0],Tet_list[i][1][0]-Tet_list[i][3][0],Tet_list[i][2][0]-Tet_list[i][3][0]],[Tet_list[i][0][1]-Tet_list[i][3][1],Tet_list[i][1][1]-Tet_list[i][3][1], Tet_list[i][2][1]-Tet_list[i][3][1]], [Tet_list[i][0][2]-Tet_list[i][3][2],Tet_list[i][1][2]-Tet_list[i][3][2], Tet_list[i][2][2]-Tet_list[i][3][2]] ]))
	return Tet_mat_4

def C_sin(k1,k2,k3):
	coeff = (1/(k3*(k2-k3)*(k1-k2)))*(np.cos(k1) - np.cos(k2)) - (1/(k3*(k2-k3)*(k1-k3)))*(np.cos(k1) - np.cos(k3)) - (1/(k2*k3*(k1-k2)))*(np.cos(k1) - np.cos(k2)) + (1/(k1*k2*k3))*np.cos(k1) - 1/(k1*k2*k3)
	return coeff

def C_cos(k1,k2,k3):
	coeff = -(1/(k3*(k2-k3)*(k1-k2)))*(np.sin(k1) - np.sin(k2)) + (1/(k3*(k2-k3)*(k1-k3)))*(np.sin(k1) - np.sin(k3)) + (1/(k2*k3*(k1-k2)))*(np.sin(k1) - np.sin(k2)) - (1/(k1*k2*k3))*np.sin(k1)
	return coeff


def Delta_FC(integer_pairs, Tets, Matrices_4):
	row = []
	for l in range(len(integer_pairs)):
		sum_S=0
		sum_C=0
		for i in range(len(Tets)):
			vec = np.matmul(np.transpose(Matrices_4[i]),integer_pairs[l])
			det = np.abs(np.linalg.det(Matrices_4[i]))
			sum_S = sum_S + det*C_sin(vec[0],vec[1],vec[2])*np.cos(np.dot(integer_pairs[l],Tets[i][3])) + det*C_cos(vec[0],vec[1],vec[2])*np.sin(np.dot(integer_pairs[l],Tets[i][3]))
			sum_C = sum_C + det*C_cos(vec[0],vec[1],vec[2])*np.cos(np.dot(integer_pairs[l],Tets[i][3])) - det*C_sin(vec[0],vec[1],vec[2])*np.sin(np.dot(integer_pairs[l],Tets[i][3]))
		row.append(sum_S)
		row.append(sum_C)

	return row


def fourier_features(tetras, ks):

    M = tetras[:, :3] - tetras[:, 3][:, np.newaxis]
    M = M.transpose(0, 2, 1)
    det_M = np.linalg.det(M)

    # mtks = np.matvec(M, ks[:, np.newaxis, :]).transpose(1, 0, 2)
    mtks = np.einsum('ijk,lj->lik', M, ks).transpose(1, 0, 2)

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
    mesh_path = join(dirname(__file__), "../../data/surfaces/shrec", "0.vtk")
    mesh = pv.read(mesh_path)


    tetras = tetrahedralize(mesh)
    k_range = 21
    ks = np.mgrid[1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1,].reshape(3, -1).T

    print(f"It starts with {mesh.n_points} points for the surface, and ends with {tetras.shape[0]} tetrahedra")

    t0 = time.time()

    M = M_getter(tetras)
    true_fcs = Delta_FC(ks, tetras, M)
    true_fcs = np.array(true_fcs).reshape((-1, 2))

    t1 = time.time()

    fcs = fourier_features(tetras, ks)

    t2 = time.time()

    print(f"Number of coeffs: {len(fcs)*2}")
    print(f"Time taken for numpy: {t1 - t0} seconds")
    print(f"Time taken for python: {t2 - t1} seconds")
    print(f"Error between both: {np.linalg.norm(true_fcs - fcs)}")

    breakpoint()


if __name__ == "__main__":
    main()

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


def fourier_features(triangles, ks):

    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(m)

    # mtks = np.matvec(mt, ks[:, np.newaxis, :]).transpose(1, 0, 2)
    mtks = np.einsum('ijk,lj->lik', mt, ks).transpose(1, 0, 2)
    a = np.power(mtks[:, :, 1] * (mtks[:, :, 0] - mtks[:, :, 1]), -1)
    b = np.power(mtks[:, :, 0] * mtks[:, :, 1], -1)
    sin_k1, sin_k2 = np.sin(mtks[:, :, 0]), np.sin(mtks[:, :, 1])
    cos_k1, cos_k2 = np.cos(mtks[:, :, 0]), np.cos(mtks[:, :, 1])
    sTks = a * (sin_k2 - sin_k1) + b * sin_k1
    cTks = a * (cos_k2 - cos_k1) + b * (cos_k1 - 1)

    cos = np.cos(np.inner(ks, triangles[:, 2])).T
    sin = np.sin(np.inner(ks, triangles[:, 2])).T
    stk = det_m[:, np.newaxis] * (sTks * cos + cTks * sin)
    ctk = det_m[:, np.newaxis] * (cTks * cos - sTks * sin)

    triangles_features = np.stack((stk, ctk), axis=-1)
    shape_features = - np.sum(triangles_features, axis=0)

    return shape_features


def M_getter(Tri_list):
	Tri_mat_4=[]
	for i in range(len(Tri_list)):
		Tri_mat_4.append(np.array([[Tri_list[i][0][0]-Tri_list[i][2][0],Tri_list[i][1][0]-Tri_list[i][2][0]],[Tri_list[i][0][1]-Tri_list[i][2][1],Tri_list[i][1][1]-Tri_list[i][2][1]]]))
	return Tri_mat_4


def C_sin(k1,k2):
	coeff = (1/(k2*(k1-k2)))*(np.sin(k2)-np.sin(k1))+(1/(k1*k2))*np.sin(k1)
	return coeff

def C_cos(k1,k2):
    coeff = -(1/(k2*(k1-k2)))*(np.cos(k1)-np.cos(k2))+(1/(k1*k2))*np.cos(k1)-(1/(k1*k2))
    return coeff

def Delta_FC(integer_pairs, Tris, Matrices_4):
	row = []
	for l in range(len(integer_pairs)):
		sum_S=0
		sum_C=0
		for i in range(len(Tris)):
			vec = np.matmul(np.transpose(Matrices_4[i]),integer_pairs[l])
			det = np.abs(np.linalg.det(Matrices_4[i]))
			sum_S = sum_S + det*C_sin(vec[0],vec[1])*np.cos(np.dot(integer_pairs[l],Tris[i][2])) + det*C_cos(vec[0],vec[1])*np.sin(np.dot(integer_pairs[l],Tris[i][2]))
			sum_C = sum_C + det*C_cos(vec[0],vec[1])*np.cos(np.dot(integer_pairs[l],Tris[i][2])) - det*C_sin(vec[0],vec[1])*np.sin(np.dot(integer_pairs[l],Tris[i][2]))
		row.append(sum_S)
		row.append(sum_C)

	return row



def main():
    leaf_data = join(dirname(__file__), "../../../data/curves/leaves", "curves.npy")
    curves = np.load(leaf_data)
    curve = curves[0] / 1.5

    triangles = triangulate_curve(curve)

    k_range = 50
    ks = np.mgrid[0 : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1].reshape(2, -1).T

    t0 = time.time()

    M = M_getter(triangles)
    true_fcs = Delta_FC(ks, triangles, M)
    true_fcs = np.array(true_fcs).reshape((-1, 2))

    t1 = time.time()

    fcs = fourier_features(triangles, ks)

    t2 = time.time()

    print(f"Number of coeffs: {len(fcs)*2}")
    print(f"Time taken for numpy: {t1 - t0} seconds")
    print(f"Time taken for python: {t2 - t1} seconds")
    true_fcs = true_fcs[~np.isnan(fcs)].reshape(-1, 2)
    fcs = fcs[~np.isnan(fcs)].reshape(-1, 2)
    print(f"Error between both: {np.linalg.norm(true_fcs - fcs)}")

    plt.title(f"The first {len(fcs)} features for a sample curve")
    plt.plot(fcs[:, 0], label="Sin")
    plt.plot(fcs[:, 1], label="Cos")

    breakpoint()


if __name__ == "__main__":
    main()

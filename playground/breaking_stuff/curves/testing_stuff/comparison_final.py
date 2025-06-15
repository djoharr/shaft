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



def fourier_features(triangles, ks):

    m = triangles[:, :2] - triangles[:, 2][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(m)

    stk, ctk = get_sc(ks, mt)

    cos = np.cos(np.inner(ks, triangles[:, 2])).T
    sin = np.sin(np.inner(ks, triangles[:, 2])).T
    sin_coefs = det_m[:, np.newaxis] * (stk * cos + ctk * sin)
    cos_coefs = det_m[:, np.newaxis] * (ctk * cos - stk * sin)

    triangles_features = np.stack((sin_coefs, cos_coefs), axis=-1)
    shape_features = - np.sum(triangles_features, axis=0)

    return shape_features


def M_getter(Tri_list):
	Tri_mat_4=[]
	for i in range(len(Tri_list)):
		Tri_mat_4.append(np.array([[Tri_list[i][0][0]-Tri_list[i][2][0],Tri_list[i][1][0]-Tri_list[i][2][0]],[Tri_list[i][0][1]-Tri_list[i][2][1],Tri_list[i][1][1]-Tri_list[i][2][1]]]))
	return Tri_mat_4


def C_sin(k1,k2):
	if k1==0 and k2!=0:
		coeff = -(1/(k2**2))*np.sin(k2) + 1/k2
	elif k2==0 and k1!=0:
		coeff = 1/k1 - (1/(k1**2))*np.sin(k1)
	elif k1==k2 and k1!=0:
		coeff = -(1/k2)*np.cos(k2) + (1/(k2**2))*np.sin(k2)
	elif k1==k2 and k1==0:
		coeff = 0
	else:
		coeff = (1/(k2*(k1-k2)))*(np.sin(k2)-np.sin(k1))+(1/(k1*k2))*np.sin(k1)
	return coeff

def C_cos(k1,k2):
	if k1 == 0 and k2!=0:
		coeff = -(1/(k2**2))-(1/(k2**2))*np.cos(k2)
	elif k2 == 0 and k1!=0:
		coeff = (1/(k1**2))-(1/(k1**2))*np.cos(k1)
	elif k1==k2 and k1!=0:
		coeff = (1/k2)*np.sin(k2) + (1/(k2**2))*np.cos(k2) - 1/(k2**2)
	elif k1==k2 and k1==0:
		coeff = 0
	else:
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

    k_range = 20
    ks = np.mgrid[0 : k_range + 0.1 : 1, 0 : k_range + 0.1 : 1,].reshape(2, -1).T

    t0 = time.time()

    M = M_getter(triangles)
    true_fcs = Delta_FC(ks, triangles, M)
    true_fcs = np.array(true_fcs).reshape((-1, 2))

    t1 = time.time()

    fcs = fourier_features(triangles, ks)

    t2 = time.time()

    mask = np.round(true_fcs, 12) == np.round(fcs, 12)
    all_missed = np.argwhere(mask[:,0]==False)


    print(f"Number of coeffs: {len(fcs)*2}")
    print(f"Time taken for numpy: {t1 - t0} seconds")
    print(f"Time taken for python: {t2 - t1} seconds")
    print(f"Error between both: {np.linalg.norm(true_fcs - fcs)}")

    # plt.title(f"The first {len(fcs)} features for a sample curve")
    # plt.plot(fcs[:, 0], label="Sin")
    # plt.plot(fcs[:, 1], label="Cos")

    breakpoint()


if __name__ == "__main__":
    main()

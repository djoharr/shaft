from os.path import join, dirname

import numpy as np
import time

from shaft.curves import triangulate_curve, curve_transform



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

def true_fc(integer_pairs, Tris, Matrices_4):
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
    leaf_data = join(dirname(__file__), "../data/curves/leaves", "curves.npy")
    curves = np.load(leaf_data)
    curve = curves[0] / 1.5

    triangles = triangulate_curve(curve)

    k_range = 20
    ks = np.mgrid[0 : k_range + 0.1 : 1, 0 : k_range + 0.1 : 1,].reshape(2, -1).T

    t0 = time.time()

    M = M_getter(triangles)
    true_fcs = true_fc(ks, triangles, M)
    true_fcs = np.array(true_fcs).reshape((-1, 2))

    t1 = time.time()

    fcs = curve_transform(triangles, ks)

    t2 = time.time()

    print(f"Number of coeffs: {len(fcs)*2}")
    print(f"Time taken for original version: {np.round(t1 - t0, 4)} seconds")
    print(f"Time taken for fast version: {np.round(t2 - t1, 4)} seconds")
    print(f"{int((t1 - t0)/(t2 - t1))}x speed up")
    print(f"Error between both: {np.linalg.norm(true_fcs - fcs)}")

    breakpoint()


if __name__ == "__main__":
    main()

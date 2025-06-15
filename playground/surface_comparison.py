from os.path import join, dirname
import numpy as np
import time
import pyvista as pv
from shaft.surfaces import tetrahedralize, surface_transform


def M_getter(Tet_list):
	Tet_mat_4=[]
	for i in range(len(Tet_list)):
		Tet_mat_4.append(np.array([[Tet_list[i][0][0]-Tet_list[i][3][0],Tet_list[i][1][0]-Tet_list[i][3][0],Tet_list[i][2][0]-Tet_list[i][3][0]],[Tet_list[i][0][1]-Tet_list[i][3][1],Tet_list[i][1][1]-Tet_list[i][3][1], Tet_list[i][2][1]-Tet_list[i][3][1]], [Tet_list[i][0][2]-Tet_list[i][3][2],Tet_list[i][1][2]-Tet_list[i][3][2], Tet_list[i][2][2]-Tet_list[i][3][2]] ]))
	return Tet_mat_4

def C_sin(k1,k2,k3):
	if k1 != 0 and k2 != 0 and k3 != 0 and k1 != k2 and k2 != k3 and k1 != k3:
		coeff = (1/(k3*(k2-k3)*(k1-k2)))*(np.cos(k1) - np.cos(k2)) - (1/(k3*(k2-k3)*(k1-k3)))*(np.cos(k1) - np.cos(k3)) - (1/(k2*k3*(k1-k2)))*(np.cos(k1) - np.cos(k2)) + (1/(k1*k2*k3))*np.cos(k1) - 1/(k1*k2*k3)
	elif k1 == 0:
		if k2 == 0 and k3 == 0:
			coeff = 0
		elif k2 ==0 and k3 != 0:
			coeff = -(1/(k3**3))*(1 - np.cos(k3)) + (1/(2*k3))
		elif k2 != 0 and k3 == 0:
			coeff = (1/(2*k2)) - (1/(k2**3))*(1 - np.cos(k2))
		else:
			if k2 == k3:
				coeff = -(1/(k2**2))*np.sin(k2) + (2/(k2**3))*(1-np.cos(k2))# + (1/(k2**2))*(1-np.cos(k2))
			else:
				coeff = -(1/(k2*k3*(k2-k3)))*(1 - np.cos(k2)) + (1/(k3*k3*(k2-k3)))*(1 - np.cos(k3)) + (1/((k2**2)*k3))*(1 - np.cos(k2))
	elif k2 == 0:
		if k1 == 0 and k3 == 0:
			coeff = 0
		elif k1 ==0 and k3 != 0:
			coeff = -(1/(k3**3))*(1 - np.cos(k3)) + (1/(2*k3))
		elif k1 != 0 and k3 == 0:
			coeff = (1/(2*k1)) + (1/(k1**3))*(np.cos(k1) - 1)
		else:
			if k1 == k3:
				coeff = -(2/(k1**3))*(np.cos(k1) - 1) - (1/(k1**2))*np.sin(k1) #- (1/(k1**2))*(np.cos(k1) - 1) - (1/(k3**2))*np.sin(k1) + (1/(k1**2))*np.cos(k1)
			else:
				coeff = (1/(k1*(k3**2)))*(1 - np.cos(k1)) + (1/((k3**2)*(k1 - k3)))*(np.cos(k1) - np.cos(k3)) - (1/((k1**2)*k3))*(np.cos(k1) - 1)# - (1/(k3*(k1**2)))*np.sin(k1) + (1/(k3*k1))*np.cos(k1)
	elif k3 == 0:
		if k1 == 0 and k2 != 0:
			coeff = (1/(2*k2)) - (1/(k2**3))*(1 - np.cos(k2))#(1/(2*k2))*np.cos(k2) - (2/(k2**2))*np.sin(k2) + 3/(2*k2)
		elif k1 == 0 and k2 == 0:
			coeff = 0
		elif k1 != 0 and k2 == 0:
			coeff = (1/(2*k1)) + (1/(k1**3))*(np.cos(k1) - 1)
		else:
			if k1 == k2:
				coeff = - (1/(k1**2))*np.sin(k1) - (2/(k1**3))*(np.cos(k1) - 1)
			else:
				coeff = -(1/((k1**2)*k2))*(np.cos(k1)-1) + (1/((k2**2)*(k1-k2)))*(np.cos(k1)-np.cos(k2)) -(1/((k2**2)*k1))*(np.cos(k1)-1)#- (1/(k1**3))*(np.cos(k1)-1) - (1/(k2*(k1**2)))*(np.cos(k1) - 1)#-(1/(k2*(k1-k2)))*(np.sin(k1) - np.sin(k2)) + (1/(k1*k2))*np.sin(k1) + (1/(k2*((k1-k2)**2)))*(np.cos(k1) - np.cos(k2)) + (1/(k2*(k1-k2)))*np.sin(k1) - (1/(k2*(k1**2)))*(np.cos(k1) - 1) - (1/(k1*k2))*np.sin(k1) + (1/((k2**2)*((k1-k2))))*(np.cos(k1) - np.cos(k2)) - (1/((k2**2)*(k1)))*(np.cos(k1)-1) + (1/((k1-k2)*(k2)))*(np.sin(k1)-np.sin(k2)) - (1/(k2*((k1-k2)**2)))*(np.cos(k1)-np.cos(k2)) - (1/(k2*((k1-k2))))*np.sin(k1)# + (1/(k1*k2))*np.sin(k1) + (1/((k1**2)*k2))*(np.cos(k1+k2) - np.cos(k2)) + (1/(k1*k2))*np.sin(k1 + k2) - (1/(k2*(k1**2)))*(np.cos(k1) - 1) - (1/(k1*k2))*np.sin(k1) + (1/(k1*(k2**2)))*(np.cos(k1+k2) - np.cos(k2)) + (1/(k1*k2))*(np.sin(k1+k2) - np.sin(k2))
	elif k1 == k2:
		if k1 == 0:
			if k3 != 0:
				coeff = -(1/(k3**3))*(1 - np.cos(k3)) + (1/(2*k3))
			else:
				coeff = 0
		else:
			if k3 == 0:
				coeff = - (1/(k2**2))*np.sin(k2) - (2/(k2**3))*(np.cos(k2) - 1)
			else:
				if k3 == k1:
					coeff = -(1/2*k1)*np.cos(k1) + (1/(k1**2))*np.sin(k1) + (1/(k1**3))*(np.cos(k1) - 1)
				else:
					coeff = -(1/(k3*(k1 - k3)))*np.sin(k1) - (1/(k3*((k1 - k3)**2)))*(np.cos(k1) - np.cos(k3)) + (1/(k1*k3))*np.sin(k1) + (1/(k1*k1*k3))*(np.cos(k1) - 1)
	elif k2 == k3:
		if k2 == 0:
			if k1 != 0:
				coeff = (1/(2*k1)) + (1/(k1**3))*(np.cos(k1) - 1)
			else:
				coeff = 0
		else:
			if k1 == 0:
				coeff = -(1/(k2**2))*np.sin(k2) + (2/(k2**3))*(1-np.cos(k2))
			else:
				if k1 == k2:
					coeff = -(1/2*k1)*np.cos(k1) + (1/(k1**2))*np.sin(k1) + (1/(k1**3))*(np.cos(k1) - 1)
				else:
					coeff = -(1/((k2*(k1-k2))))*(np.sin(k1)-np.sin(k2)) + (1/(k2*((k1-k2)**2)))*(np.cos(k1) - np.cos(k2)) + (1/((k2)*(k1-k2)))*np.sin(k1) - (1/((k2**2)*(k1-k2)))*(np.cos(k1) - np.cos(k2)) + (1/((k2**2)*k1))*(np.cos(k1)-1)
	elif k1 == k3:
		if k1 == 0:
			if k2 != 0:
				coeff = (1/(2*k2)) - (1/(k2**3))*(1-np.cos(k2))
			else:
				coeff = 0
		else:
			if k2 == 0:
				coeff = -(2/(k1**3))*(np.cos(k1) - 1) - (1/(k1**2))*np.sin(k1) #- (1/(k1**2))*(np.cos(k1) - 1) - (1/(k3**2))*np.sin(k1) + (1/(k1**2))*np.cos(k1)
			else:
				if k2 == k1:
					coeff = -(1/(2*k1))*np.cos(k1) + (1/(k1**2))*np.sin(k1) + (1/(k1**3))*(np.cos(k1) - 1)
				else:
					coeff = -(1/(k1*((k2-k1)**2)))*(np.cos(k1) - np.cos(k2)) - (1/(k2*k1*(k1 - k2)))*(np.cos(k1) - np.cos(k2)) + (1/(k2*(k1**2)))*(np.cos(k1) - 1) + (1/(k1*(k2 - k1)))*np.sin(k1)
	return coeff

def C_cos(k1,k2,k3):
	if k1 != 0 and k2 != 0 and k3 != 0 and k1 != k2 and k2 != k3 and k1 != k3:
		coeff = -(1/(k3*(k2-k3)*(k1-k2)))*(np.sin(k1) - np.sin(k2)) + (1/(k3*(k2-k3)*(k1-k3)))*(np.sin(k1) - np.sin(k3)) + (1/(k2*k3*(k1-k2)))*(np.sin(k1) - np.sin(k2)) - (1/(k1*k2*k3))*np.sin(k1)
	elif k1 == 0:
		if k2 == 0 and k3 != 0:
			coeff = (1/(k3**2)) - (1/(k3**3))*np.sin(k3)
		elif k2 == 0 and k3 == 0:
			coeff = 0
		elif k2 != 0 and k3 == 0:
			coeff = (1/(k2**2)) - (1/(k2**3))*np.sin(k2)
		else:
			if k2 == k3:
				coeff = (1/(k2**2))*(1-np.cos(k2)) + (2/(k2**3))*np.sin(k2) - (2/(k2**2))# - (1/(k3**3))
			else:
				coeff = -(1/(k2*k3*(k2 - k3)))*np.sin(k2) + (1/((k3**2)*(k2 - k3)))*np.sin(k3) + (1/(k3*(k2**2)))*np.sin(k2) - 1/(k2*k3)
	elif k2 == 0:
		if k1 == 0 and k3 != 0:
			coeff = (1/(k3**2)) - (1/(k3**3))*np.sin(k3)
		elif k1 == 0 and k3 == 0:
			coeff = 0
		elif k1 != 0 and k3 == 0:
			coeff = (1/(k1**2)) - (1/(k1**3))*(np.sin(k1))
		else:
			if k1 == k3:
				coeff = (2/(k1**3))*np.sin(k1) - (1/(k1**2)) - (1/(k1**2))*np.cos(k1)
			else:
				coeff = -(1/((k3**2)*(k1 - k3)))*(np.sin(k1) - np.sin(k3)) + (1/((k3**2)*(k1)))*np.sin(k1)+ (1/(k3*(k1**2)))*np.sin(k1) + (1/(k3*k1))*(np.cos(k1) - 1)  - (1/(k1*k3))*np.cos(k1)
	elif k3 == 0:
		if k1 == 0 and k2 != 0:
			coeff = - (1/(k2**3))*np.sin(k2) + (1/(k2**2))
		elif k1 == 0 and k2 == 0:
			coeff = 0
		elif k1 != 0 and k2 == 0:
			coeff = (1/(k1**2)) - (1/(k1**3))*(np.sin(k1))
		else:
			if k1 == k2:
				coeff = (1/(k1**2))*(np.cos(k1)-1) + (2/(k1**3))*np.sin(k1) - (2/(k1**2))*np.cos(k1)
			else:
				coeff =  - (1/(k1*k2)) + (1/(k1*(k2**2)))*np.sin(k1) + (1/(k2*(k1**2)))*np.sin(k1) - (1/((k2**2)*(k1-k2)))*(np.sin(k1)-np.sin(k2))# - (1/(k2*((k1-k2)**2)))*(np.sin(k1)-np.sin(k2)) + (1/(k2*(k1-k2)))*np.cos(k1) + ((1/(k2*(k1**2)))+(1/(k1*(k2**2))))*np.sin(k1) - (1/((k2**2)*(k1-k2)))*(np.sin(k1)-np.sin(k2))#+ (1/((k2**2)*(k1-k2)))*np.cos(k1) + (1/((k1**2)*k2))*np.sin(k1) - (1/(k2*k1))*np.cos(k1) - (1/((k2**2)*(k1-k2)))*(np.sin(k1)-np.sin(k2)) + (1/((k1*(k2**2))))*np.sin(k1) - (1/(k2*(k1-k2)))*(np.cos(k1)-np.cos(k2)) - (1/(k2*(k1-k2)))*np.cos(k1)
	elif k1 == k2:
		if k1 == 0:
			if k3 != 0:
				coeff = (1/(k3**2)) - (1/(k3**3))*np.sin(k3)
			else:
				coeff = 0
		else:
			if k3 == 0:
				coeff = (1/(k1**2))*(np.cos(k1)-1) + (2/(k1**3))*np.sin(k1) - (2/(k1**2))*np.cos(k1)
			else:
				if k3 == k1:
					coeff = (1/(2*k1))*np.sin(k1) + (1/(k1**2))*np.cos(k1) - (1/(k1**3))*np.sin(k1)
				else:
					coeff = -(1/((k3)*(k1-k3)))*np.cos(k1) + (1/(k3*((k1-k3)**2)))*(np.sin(k1) - np.sin(k3)) + (1/(k3*k1))*np.cos(k1) - (1/(k3*(k1**2)))*np.sin(k1)
	elif k2 == k3:
		if k2 == 0:
			if k1 != 0:
				coeff = (1/(k1**2)) - (1/(k1**3))*(np.sin(k1))
			else:
				coeff = 0
		else:
			if k1 == 0:
				coeff = (1/(k2**2))*(1-np.cos(k2)) + (2/(k2**3))*np.sin(k2) - (2/(k2**2))
			else:
				if k1 == k2:
					coeff = (1/(2*k1))*np.sin(k1) + (1/(k1**2))*np.cos(k1) - (1/(k1**3))*np.sin(k1)
				else:
					coeff = -(1/(k2*(k1-k2)))*(np.cos(k1) - np.cos(k2)) - (1/(k2*((k1-k2)**2)))*(np.sin(k1)-np.sin(k2)) + (1/(k2*(k1-k2)))*np.cos(k1) + (1/((k1-k2)*(k2**2)))*(np.sin(k1)-np.sin(k2)) - (1/(k1*(k2**2)))*np.sin(k1)
	elif k1 == k3:
		if k1 == 0:
			if k2 != 0:
				coeff = (1/(k2**2)) - (1/(k2**3))*np.sin(k3)
			else:
				coeff = 0
		else:
			if k2 == 0:
				coeff = (2/(k3**3))*np.sin(k3) - (1/(k3**2))*np.cos(k3) - (1/(k3**2))
			else:
				if k2 == k1:
					coeff = (1/(2*k1))*np.sin(k1) + (1/(k1**2))*np.cos(k1) - (1/(k1**3))*np.sin(k1)
				else:
					coeff = (1/(k1*((k1-k2)**2)))*(np.sin(k1)-np.sin(k2)) - (1/(k1*(k1-k2)))*np.cos(k1) + (1/(k1*k2*(k1-k2)))*(np.sin(k1)-np.sin(k2)) - (1/(k2*(k1**2)))*np.sin(k1)
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



def main():
    mesh_path = join(dirname(__file__), "../data/surfaces/shrec", "0.vtk")
    mesh = pv.read(mesh_path)


    tetras = tetrahedralize(mesh)
    k_range = 10
    ks = np.mgrid[1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1,].reshape(3, -1).T

    print(f"It starts with {mesh.n_points} points for the surface, and ends with {tetras.shape[0]} tetrahedra")

    t0 = time.time()

    M = M_getter(tetras)
    true_fcs = Delta_FC(ks, tetras, M)
    true_fcs = np.array(true_fcs).reshape((-1, 2))

    t1 = time.time()

    fcs = surface_transform(tetras, ks)

    t2 = time.time()

    fcs_not = surface_transform(tetras, ks, taylor=False)

    t3 = time.time()

    print(f"Number of coeffs: {len(fcs)*2}")
    print(f"Time taken for python: {np.round(t1 - t0, 4)} seconds")
    print(f"Time taken for numpy: {np.round(t2 - t1, 4)} seconds")
    print(f"{int((t1 - t0)/(t2 - t1))}x speed up")
    print(f"Error between both: {np.linalg.norm(true_fcs - fcs)}")
    print(f"Time taken w/o taylor: {np.round(t3 - t2, 4)} seconds")
    print(f"{np.round((t2 - t1)/(t3 - t2), 2)}x speed down")
    breakpoint()


if __name__ == "__main__":
    main()

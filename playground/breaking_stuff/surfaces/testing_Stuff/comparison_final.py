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

def get_masks(ks):
    nx, ny, nz = ks[:, :, 0] == 0, ks[:, :, 1] == 0, ks[:, :, 2] == 0
    exy, exz, eyz = ks[:,:,0]==ks[:,:,1], ks[:,:,0]==ks[:,:,2], ks[:,:,1]==ks[:,:,2]
    nxyz = nx & ny & nz
    nxy = nx & ny & ~nxyz
    nxz = nx & nz & ~nxyz
    nyz = ny & nz & ~nxyz
    exynz = exy & nz & ~nxyz
    exy = exy & ~nx & ~ny & ~nz & ~exynz
    exzny = exz & ny & ~nxyz
    exz = exz & ~nx & ~ny & ~nz & ~exzny
    eyznx = eyz & nx & ~nxyz
    eyz = eyz & ~nx & ~ny & ~nz & ~eyznx
    exyz = exy & eyz
    exy = exy & ~exyz
    exz = exz & ~exyz
    eyz = eyz & ~exyz
    nx = nx & ~nxy & ~nxz & ~eyznx & ~nxyz
    ny = ny & ~nxy & ~nyz & ~exzny & ~nxyz
    nz = nz & ~nxz & ~nyz & ~exynz & ~nxyz

    rest = ~nx & ~ny & ~nz & ~nxy & ~nxz & ~nyz & ~nxyz
    rest = rest & ~exy & ~exz & ~eyz & ~exyz & ~exynz & ~exzny & ~eyznx

    return rest, nx, ny, nz, nxy, nxz, nyz, nxyz, exy, exz, eyz, exyz, exynz, exzny, eyznx


def get_sc(ks, mt):

    mat_base = np.einsum('ijk,lj->lik', mt, ks)
    sins = np.zeros(mat_base.shape[:2])
    coss = np.zeros(mat_base.shape[:2])

    normies, nx, ny, nz, nxy, nxz, nyz, nxyz, exy, exz, eyz, exyz, exynz, exzny, eyznx = get_masks(mat_base)

    mnorm = mat_base[normies]
    x, y, z = mnorm[:, 0], mnorm[:, 1], mnorm[:, 2]
    sinx, siny, sinz = np.sin(x), np.sin(y), np.sin(z)
    cosx, cosy, cosz = np.cos(x), np.cos(y), np.cos(z)
    a, b = np.power(z * (y-z) * (x-y), -1), np.power(z * (y-z) * (x-z), -1)
    c, d = np.power(y * z * (x-y), -1), np.power(x * y * z, -1)
    sins[normies] = a * (cosx - cosy) - b * (cosx - cosz) - c * (cosx - cosy) + d * (cosx - 1)
    coss[normies] = - a * (sinx - siny) + b * (sinx - sinz) + c * (sinx - siny) - d * sinx

    mnx = mat_base[nx]
    y, z = mnx[:, 1], mnx[:, 2]
    siny, sinz, cosy, cosz = np.sin(y), np.sin(z), np.cos(y), np.cos(z)
    a, b = np.power(y*z*(y-z), -1), np.power(z**2*(y-z), -1)
    c, d = np.power(z*y**2, -1), np.power(y*z, -1)
    sins[nx] = (c - a) * (1 - cosy) + b * (1 - cosz)
    coss[nx] = (c - a) * siny + b * sinz - d

    mny = mat_base[ny]
    x, z = mny[:, 0], mny[:, 2]
    sinx, sinz, cosx, cosz = np.sin(x), np.sin(z), np.cos(x), np.cos(z)
    a, b = np.power(x*z**2, -1), np.power(z**2*(x-z), -1)
    c, d = np.power(z*x**2, -1), np.power(x*z, -1)
    sins[nx] = (a + c)*(1 - cosx) + b*(cosx - cosz)
    coss[nx] = (a + c) * sinx - b * (sinx - sinz) - d

    mnz = mat_base[nz]
    x, y = mnz[:, 0], mnz[:, 1]
    sinx, siny, cosx, cosy = np.sin(x), np.sin(y), np.cos(x), np.cos(y)
    a, b = np.power(y*x**2, -1), np.power(y**2*(x-y), -1)
    c, d = np.power(x*y**2, -1), np.power(x*y, -1)
    sins[nz] = (a + c) * (1 - cosx) + b * (cosx - cosy)
    coss[nz] = (a + c) * sinx - b * (sinx - siny) - d

    mnxy = mat_base[nxy]
    z = mnxy[:, 2]
    sinz, cosz = np.sin(z), np.cos(z)
    a, b, c = np.power(z, -3), np.power(2*z, -1), np.power(z, -2)
    sins[nxy] = a*(cosz - 1) + b
    coss[nxy] = c - a*sinz

    mnxz = mat_base[nxz]
    y = mnxz[:, 1]
    siny, cosy = np.sin(y), np.cos(y)
    a, b, c = np.power(2*y, -1), np.power(y, -3), np.power(y, -2)
    sins[nxz] = a - b * (1 - cosy)
    coss[nxz] = c - b * siny

    mnyz = mat_base[nyz]
    x = mnyz[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b, c = np.power(2*x, -1), np.power(x, -3), np.power(x, -2)
    sins[nyz] = a + b * (cosx - 1)
    coss[nyz] = c - b * sinx

    mnxyz = mat_base[nxyz]
    sins[nxyz] = 0
    coss[nxyz] = 0

    mexy = mat_base[exy]
    x, z = mexy[:, 0], mexy[:, 2]
    sinx, sinz, cosx, cosz = np.sin(x), np.sin(z), np.cos(x), np.cos(z)
    a, b = np.power(z*(x-z), -1), np.power(z*(x-z)**2, -1)
    c, d = np.power(x*z, -1), np.power(z*x**2, -1)
    sins[exy] = -a*sinx - b*(cosx - cosz) + c*sinx + d*(cosx - 1)
    coss[exy] = -a*cosx + b*(sinx - sinz) + c*cosx - d*sinx

    mexz = mat_base[exz]
    x, y = mexz[:, 0], mexz[:, 1]
    sinx, siny, cosx, cosy = np.sin(x), np.sin(y), np.cos(x), np.cos(y)
    a, b = np.power(x*(y-x), -1), np.power(x*(y-x)**2, -1)
    c, d, e = np.power(y*x*(x-y), -1), np.power(y*x**2, -1), np.power(x*(x-y)**2, -1)
    sins[exz] = a*sinx - (b + c)*(cosx - cosy) + d*(cosx - 1)
    coss[exz] = e*(sinx - siny) - a*cosx + c*(sinx - siny) - d*sinx

    meyz = mat_base[eyz]
    x, y = meyz[:, 0], meyz[:, 1]
    sinx, siny, cosx, cosy = np.sin(x), np.sin(y), np.cos(x), np.cos(y)
    a, b = np.power(y*(x-y), -1), np.power(y*(x-y)**2, -1)
    c, d = np.power((x-y)*y**2, -1), np.power(x*y**2, -1)
    sins[eyz] = a*siny + b*(cosx - cosy) - c*(cosx - cosy) + d*(cosx-1)
    coss[eyz] = -a*cosy - b*(sinx - siny) + c*(sinx - siny) - d*sinx

    mexyz = mat_base[exyz]
    x = mexyz[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b, c = np.power(2*x, -1), np.power(x, -2), np.power(x, -3)
    sins[exyz] = -a*cosx + b*sinx + c*(cosx - 1)
    coss[exyz] = a*sinx + b*cosx - c*sinx

    mexynz = mat_base[exynz]
    x = mexynz[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b = np.power(x, -2), np.power(x, -3)
    sins[exynz] = -a * sinx - b * (cosx - 1)
    coss[exynz] = a * (cosx - 1) + 2 * b * sinx - 2 * a * cosx

    mexzny = mat_base[exzny]
    x = mexzny[:, 0]
    sinx, cosx = np.sin(x), np.cos(x)
    a, b = np.power(x, -3), np.power(x, -2)
    sins[exzny] = -2 * a * (cosx - 1) - b * sinx
    coss[exzny] = 2 * a * sinx - b * (cosx - 1)

    meyznx = mat_base[eyznx]
    y = meyznx[:, 1]
    siny, cosy = np.sin(y), np.cos(y)
    a, b = np.power(y, -2), np.power(y, -3)
    sins[eyznx] = -a * siny + 2 * b * (1 - cosy)
    coss[eyznx] = -a * (1 + cosy) + 2 * b * siny

    return sins.T, coss.T

def get_sc_no_taylor(ks, mt):
    mat_base = np.einsum('ijk,lj->lik', mt, ks).transpose(1, 0, 2)
    x, y, z = mat_base[:, :, 0], mat_base[:, :, 1], mat_base[:, :, 2]
    a = np.power(z * (y-z) * (x-y), -1)
    b = np.power(z * (y-z) * (x-z), -1)
    c = np.power(y * z * (x-y), -1)
    d = np.power(x * y * z, -1)
    sinx, siny, sinz = np.sin(x), np.sin(y), np.sin(z)
    cosx, cosy, cosz = np.cos(x), np.cos(y), np.cos(z)
    sins = - a * (sinx - siny) + b * (sinx - sinz) + c * (sinx - siny) - d * sinx
    coss = a * (cosx - cosy) - b * (cosx - cosz) - c * (cosx - cosy) + d * (cosx - 1)
    return sins, coss

def fourier_features(tetras, ks, taylor=True):

    m = tetras[:, :3] - tetras[:, 3][:, np.newaxis]
    mt = m.transpose(0, 2, 1)
    det_m = np.linalg.det(mt)

    if taylor:
        stk, ctk = get_sc(ks, mt)
    else:
        stk, ctk = get_sc_no_taylor(ks, mt)

    cos = np.cos(np.inner(ks, tetras[:, 3])).T
    sin = np.sin(np.inner(ks, tetras[:, 3])).T
    sin_coefs = det_m[:, np.newaxis] * (stk * cos + ctk * sin)
    cos_coefs = det_m[:, np.newaxis] * (ctk * cos - stk * sin)

    tetras_features = np.stack((sin_coefs, cos_coefs), axis=-1)
    shape_features = - np.sum(tetras_features, axis=0)

    return shape_features


def main():
    mesh_path = join(dirname(__file__), "../../../data/surfaces/shrec", "0.vtk")
    mesh = pv.read(mesh_path)


    tetras = tetrahedralize(mesh)
    k_range = 50
    ks = np.mgrid[1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1, 1 : k_range + 0.1 : 1,].reshape(3, -1).T

    print(f"It starts with {mesh.n_points} points for the surface, and ends with {tetras.shape[0]} tetrahedra")



    # ks = np.mgrid[-k_range : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1, -k_range : k_range + 0.1 : 1,].reshape(3, -1).T
    # m = np.array([np.eye(3) for i in range(len(tetras))])
    # m = tetras[:, :3] - tetras[:, 3][:, np.newaxis]
    # mt = m.transpose(0, 2, 1)
    # stk, ctk = get_sc(ks, mt)

    t0 = time.time()

    # M = M_getter(tetras)
    # true_fcs = Delta_FC(ks, tetras, M)
    # true_fcs = np.array(true_fcs).reshape((-1, 2))

    t1 = time.time()

    fcs = fourier_features(tetras, ks)

    t2 = time.time()

    fcs_not = fourier_features(tetras, ks, taylor=False)

    t3 = time.time()

    # mask = np.round(true_fcs, 12) == np.round(fcs, 12)
    # all_missed = np.argwhere(mask[:,0]==False)

    mask_taylor = np.round(fcs_not, 12) == np.round(fcs, 12)
    all_missed_taylor = np.argwhere(mask_taylor[:,0]==False)

    print(f"Number of coeffs: {len(fcs)*2}")
    print(f"Time taken for python: {t1 - t0} seconds")
    print(f"Time taken for numpy: {t2 - t1} seconds")
    print(f"Time taken w/o taylor: {t3 - t2} seconds")
    # print(f"Error between both: {np.linalg.norm(true_fcs - fcs)}")
    print(f"Error for taylor: {np.linalg.norm(fcs - fcs_not)}")

    breakpoint()


if __name__ == "__main__":
    main()

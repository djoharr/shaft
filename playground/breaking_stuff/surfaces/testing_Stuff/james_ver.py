#from os.path import join, dirname
import os
import numpy as np
import scipy as sp
import pandas as pd
import xlsxwriter
import pyvista as pv
import matplotlib.pyplot as plt
import tetgen


########################################################################
###################### GET TETRA COORDS ################################
########################################################################

def tetrahedralize(surface):
	tet = tetgen.TetGen(surface)
	points, tets = tet.tetrahedralize()
	tetras = points[tets]
	l = list(tetras.flatten())
	t_l = int(len(l)/3)
	new_tets = [l[0 + 3*i:3 + 3*i] for i in range(0, t_l)]
	#print(new_tris)
	t_l = int(len(new_tets)/4)
	new_tetra_coords = [new_tets[0 + 4*i: 4 + 4*i] for i in range(0,t_l)]
	return new_tetra_coords

########################################################################
################# GENERATING LATTICE POINTS ############################
########################################################################
#size = 7
integer_trips = np.mgrid[1:7+0.1:1, 1:7+0.1:1, 1:7+0.1:1].reshape(3,-1).T

col_names=[]
for i in range(len(integer_trips)):
    col_names.append('S('+np.array2string(integer_trips[i][0])+np.array2string(integer_trips[i][1])+np.array2string(integer_trips[i][2])+')')
    col_names.append('C('+np.array2string(integer_trips[i][0])+np.array2string(integer_trips[i][1])+np.array2string(integer_trips[i][2])+')')

Fourier_table = pd.DataFrame(columns = col_names)

########################################################################
########################### GET M MATRICES #############################
########################################################################
def M_getter(Tet_list):
	Tet_mat_4=[]
	for i in range(len(Tet_list)):
		Tet_mat_4.append(np.array([[Tet_list[i][0][0]-Tet_list[i][3][0],Tet_list[i][1][0]-Tet_list[i][3][0],Tet_list[i][2][0]-Tet_list[i][3][0]],[Tet_list[i][0][1]-Tet_list[i][3][1],Tet_list[i][1][1]-Tet_list[i][3][1], Tet_list[i][2][1]-Tet_list[i][3][1]], [Tet_list[i][0][2]-Tet_list[i][3][2],Tet_list[i][1][2]-Tet_list[i][3][2], Tet_list[i][2][2]-Tet_list[i][3][2]] ]))
	return Tet_mat_4

########################################################################
############### CALCULATING FOURIER COEFFICIENTS #######################
########################################################################
#Each shape comes with a set of triangles approximating it, and each set of triangles comes with a 
#set of matrices for calculating the Fourier coefficient of each matrix. The Fourier coefficient of the
#shape is given by the sum of the Fourier coefficients for each triangle. We do this here.
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
	row = [] #set up a row holding the Fourier Coefficients of that shape
	for l in range(len(integer_pairs)): #Calculate the Fourier Coefficient for each pair of integers in the lattice
		sum_S=0
		sum_C=0
		for i in range(len(Tets)):
			vec = np.matmul(np.transpose(Matrices_4[i]),integer_pairs[l])
			det = np.abs(np.linalg.det(Matrices_4[i]))
			sum_S = sum_S + det*C_sin(vec[0],vec[1],vec[2])*np.cos(np.dot(integer_pairs[l],Tets[i][3])) + det*C_cos(vec[0],vec[1],vec[2])*np.sin(np.dot(integer_pairs[l],Tets[i][3]))
			sum_C = sum_C + det*C_cos(vec[0],vec[1],vec[2])*np.cos(np.dot(integer_pairs[l],Tets[i][3])) - det*C_sin(vec[0],vec[1],vec[2])*np.sin(np.dot(integer_pairs[l],Tets[i][3]))
		row.append(sum_S) #Append the FC to the row
		row.append(sum_C)

	return row

########################################################################
################ GENERATE 1-PARAMETER FAMILY OF SHAPES #################
########################################################################
#for folder, subfolders, files in os.walk('shrec'):
	#if folder == 'shrec':
	#if subfolders == 'VTK_files':
vtk_files = []
for filename in os.listdir('shrec'):
	file = os.path.join('shrec', filename)
	vtk_files.append(file)
#vtk_docs = [f for f in files if f.endswith(".vtk")]
#lst = []
#	print(vtk_docs)
#for vtk in vtk_docs:
#lst.append(os.path.join(folder,vtk))
#print(lst)
shit_lst = []
#	size = 1
#	integer_trips = np.mgrid[-size:size+0.1:1, -size:size+0.1:1, -size:size+0.1:1].reshape(3,-1).T
#	col_names=[]
#	for i in range(len(integer_trips)):
#		col_names.append('S('+np.array2string(integer_trips[i][0])+np.array2string(integer_trips[i][1])+np.array2string(integer_trips[i][2])+')')
#		col_names.append('C('+np.array2string(integer_trips[i][0])+np.array2string(integer_trips[i][1])+np.array2string(integer_trips[i][2])+')')
#	Fourier_table = pd.DataFrame(columns = col_names)
for vtk in vtk_files:
	try:
		Tetras = tetrahedralize(vtk)
		M = M_getter(Tetras)
		row = Delta_FC(integer_trips, Tetras, M)
		Fourier_table.loc[len(Fourier_table)] = row
	except:
		shit_lst.append(vtk)

df_info = pd.read_csv("shrec_info.csv")
FT = pd.concat([Fourier_table,df_info['class']],axis=1)
#print(len(FT.loc[FT.isnull().any(axis=1)]))
FT_clean = FT.dropna()
writer = pd.ExcelWriter('Shrec_SHAFT.xlsx', engine = 'xlsxwriter')
workbook = writer.book
worksheet  = workbook.add_worksheet('Fourier Coefficients')
writer.sheets['Fourier Coefficients'] = worksheet
#worksheet.write_string(0, 0, 'Variables with 0 variance')
#
FT_clean.to_excel(writer, sheet_name = 'Fourier Coefficients', startrow = 0, startcol = 0)
writer.close()

#print(len(shit_lst), len(vtk_docs))
#mesh_path = "0.vtk"
#mesh = pv.read(mesh_path)
#tetras = tetrahedralize(mesh)


#print(tetras)
#mesh.plot(show_edges=True, line_width = 5)


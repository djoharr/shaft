from os.path import join, dirname

import numpy as np
import scipy as sp
import pandas as pd
import xlsxwriter
import pyvista as pv
import matplotlib.pyplot as plt


########################################################################
#################### LOAD DATA #########################################
#######################################################################
df_info = pd.read_csv("info_shells.csv")

#leaf_data = join(dirname(__file__), "Users\caram\Documents\Coding\data\curves\leaves\", "curves.npy")
curves = np.load("curves_shells.npy")
#curves = np.load("C:\Users\caram\Documents\Coding\data\curves\leaves\curves.npy")
#print(curves[0])
#curve_0 = curves[0] / 1.5
#curve_1 = curves[1] / 1.5

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
	l = list(triangles.flatten())
	t_l = int(len(l)/2)
	new_tris = [l[0 + 2*i:2 + 2*i] for i in range(0, t_l)]
	#print(new_tris)
	t_l = int(len(new_tris)/3)
	new_tris_coords = [new_tris[0 + 3*i: 3 + 3*i] for i in range(0,t_l)]
	return new_tris_coords

####################################################################################################
################################# GENERATING LATTICE POINTS #################################
####################################################################################################
#ks = np.mgrid[0 :  + 0.1 : 1, -k_range : k_range + 0.1 : 1].reshape(2, -1).T
# Really crude way of doing this, but the ordering of the coefficients doesn't matter so much and will be picked out by the statistics later, anyway.
#integer_pairs = []
#for i in range(3):
#    for j in range(3):
#        k=np.array([0+i,0+j])
#        integer_pairs.append(k)

#Reflect these points across the axes to obtain all lattice point up to 5
#for i in range(len(integer_pairs)):
#    k1 = np.array([integer_pairs[i][0],-integer_pairs[i][1]])
#    integer_pairs.append(k1)
#    k2 = np.array([-integer_pairs[i][0],integer_pairs[i][1]])
#    integer_pairs.append(k2)
#    k3 = np.array([-integer_pairs[i][0],-integer_pairs[i][1]])
#    integer_pairs.append(k3)

#integer_pairs = np.mgrid[-7 : 7 + 0.1 : 1, -7 : 7 + 0.1 : 1].reshape(2, -1).T

#col_names=[]
#for i in range(len(integer_pairs)):
#    col_names.append('S('+np.array2string(integer_pairs[i][0])+np.array2string(integer_pairs[i][1])+')')
#    col_names.append('C('+np.array2string(integer_pairs[i][0])+np.array2string(integer_pairs[i][1])+')')

#Fourier_table = pd.DataFrame(columns = col_names)

#SPIDER WEB GRID
sample_points = []
inner_radius = 3
outer_radius = 7
slices = 6
for i in range(inner_radius, outer_radius):
	for j in range(0,slices):
		sample_points.append([i*np.cos(((2*np.pi)/slices)*j), i*np.sin(((2*np.pi)/slices)*j)])

col_names=[]
for i in range(len(sample_points)):
    col_names.append('S('+np.array2string(sample_points[i][0])+np.array2string(sample_points[i][1])+')')
    col_names.append('C('+np.array2string(sample_points[i][0])+np.array2string(sample_points[i][1])+')')
integer_pairs = sample_points
Fourier_table = pd.DataFrame(columns = col_names)		

########################################################################
################## GET M MATRICES ######################################
########################################################################
def M_getter(Tri_list):
	Tri_mat_4=[]
	for i in range(len(Tri_list)):
		Tri_mat_4.append(np.array([[Tri_list[i][0][0]-Tri_list[i][2][0],Tri_list[i][1][0]-Tri_list[i][2][0]],[Tri_list[i][0][1]-Tri_list[i][2][1],Tri_list[i][1][1]-Tri_list[i][2][1]]]))
	return Tri_mat_4

####################################################################################################
################################# CALCULATING FOURIER COEFFICIENTS #################################
####################################################################################################
#Each shape comes with a set of triangles approximating it, and each set of triangles comes with a 
#set of matrices for calculating the Fourier coefficient of each matrix. The Fourier coefficient of the
#shape is given by the sum of the Fourier coefficients for each triangle. We do this here.
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
	row = [] #set up a row holding the Fourier Coefficients of that shape
	for l in range(len(integer_pairs)): #Calculate the Fourier Coefficient for each pair of integers in the lattice
		sum_S=0
		sum_C=0
		for i in range(len(Tris)):
			vec = np.matmul(np.transpose(Matrices_4[i]),integer_pairs[l])
			det = np.abs(np.linalg.det(Matrices_4[i]))
			sum_S = sum_S + det*C_sin(vec[0],vec[1])*np.cos(np.dot(integer_pairs[l],Tris[i][2])) + det*C_cos(vec[0],vec[1])*np.sin(np.dot(integer_pairs[l],Tris[i][2]))
			sum_C = sum_C + det*C_cos(vec[0],vec[1])*np.cos(np.dot(integer_pairs[l],Tris[i][2])) - det*C_sin(vec[0],vec[1])*np.sin(np.dot(integer_pairs[l],Tris[i][2]))
		row.append(sum_S) #Append the FC to the row
		row.append(sum_C)

	return row
########################################################################
################ GENERATE 1-PARAMETER FAMILY OF SHAPES #################
########################################################################


for j in range(0,len(curves)):
	Tris = triangulate_curve(curves[j])
	M = M_getter(Tris)
	row = Delta_FC(integer_pairs, Tris, M)
	Fourier_table.loc[len(Fourier_table)] = row


FT = pd.concat([Fourier_table,df_info['genusName']],axis=1)
#print(len(FT.loc[FT.isnull().any(axis=1)]))
FT_clean = FT.dropna()
writer = pd.ExcelWriter('Shells_SHAFT.xlsx', engine = 'xlsxwriter')
workbook = writer.book
worksheet  = workbook.add_worksheet('Fourier Coefficients')
writer.sheets['Fourier Coefficients'] = worksheet
#worksheet.write_string(0, 0, 'Variables with 0 variance')
#
FT_clean.to_excel(writer, sheet_name = 'Fourier Coefficients', startrow = 0, startcol = 0)
writer.close()

#print(FT_clean)

#######################################################################
###################### REP FROM EACH GROUP ############################
#######################################################################

#GROUP 1

cp = np.array(curves[1])
x = cp[:,0]
y = cp[:,1]
fig, ax = plt.subplots()
ax.plot(x,y)
plt.savefig("shell1.png")

#GROUP 2
cp = np.array(curves[5])
x = cp[:,0]
y = cp[:,1]
fig, ax = plt.subplots()
ax.plot(x,y)
plt.savefig("shell2.png")

#GROUP 3
#curve_points = []
#for m in range(k):
#	curve_points.append([(1+axes3[0]*np.sin(((2*np.pi)/k)*m))*np.cos(((2*np.pi)/k)*m),(1+axes3[0]*np.sin(3*((2*np.pi)/k)*m))*np.sin(((2*np.pi)/k)*m)])
#cp = np.array(curve_points)
#x = cp[:,0]
#y = cp[:,1]
#fig, ax = plt.subplots()
#ax.plot(x,y)
#plt.savefig("group3.png")



#k_range = 200




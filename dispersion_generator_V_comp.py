# Date: 06/16/2023
# Amir Husen

import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, isclose
import h5py
import os, sys, yaml
import pandas as pd
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import time

#t1 = time.time()

# Remember to modify POSCAR file to correct lattice parameter
root = '2_6_50'
#ndisps = 1 # Number of dispersions to generate (equal to number of yalm files)
n = 250 # Number of atoms
#init_no = 0
system_size = 5
a_val = 3.03 #Side length of each cube in bcc
alat = a_val * system_size # the size of the box



#Generating atoms position
positions = []

basis = np.eye(3)*a_val
base_atoms = np.array([[0.0, 0.0, 0.0]])*a_val
for i in range(system_size): 
    for j in range(system_size):
        for k in range(system_size):
            base_position = np.array([k, j, i]) #Local variable
            cart_position = np.inner(basis.T, base_position) #Local
            for atom in base_atoms:
                positions.append(cart_position + atom)
base_atoms = np.array([[0.5, 0.5, 0.5]])*a_val
for i in range(system_size): 
    for j in range(system_size):
        for k in range(system_size):
            base_position = np.array([k, j, i]) #Local variable
            cart_position = np.inner(basis.T, base_position) #Local
            for atom in base_atoms:
                positions.append(cart_position + atom)
ide_lat = np.array(positions)

#print(ide_lat.shape)
#t2 = time.time()
#print(t2 - t1)
#sys.exit()



ideal_distances = np.zeros((n, 3, n))
ideal_dist_sca = np.zeros((n, n)) # sca stands for scalar

for i in range(n):
    for j in range(n):
        if i != j:
            ideal_distances[i, :, j] = ide_lat[j, :] - ide_lat[i, :] #Ideal distance vector from i to j

            #Apply mic: 
            for d in range(3):
                if ideal_distances[i, d, j] > (alat / 2):
                    ideal_distances[i, d, j] -= alat

                elif ideal_distances[i, d, j] <= (-alat / 2):
                    ideal_distances[i, d, j] += alat

            ideal_dist_sca[i, j] = np.linalg.norm(ideal_distances[i, :, j]) #Distance scalar
#print(ideal_dist_sca[:, :])
#sys.exit()

neighbors = [] #List of dictionaries holding a list of the IDs for all types of nn

#Distances to each type of neighbor:
first = (sqrt(3) * a_val) / 2
second = a_val
third = sqrt(2) * a_val
fourth = (sqrt(11) * a_val) / 2
fifth = sqrt(3) * a_val
nn_dist = [first, second, third, fourth, fifth]

for i in range(n):
    i_neigh = [[i]] 
    for prox in range(5):
        i_neigh.append([]) #Empty placeholder lists for neighbors of i
    
    #Add each list of nn to the placeholder dictionary, append the dictionary to the list:
    for j in range(n):
        if i != j:
            for prox in range(1,6):
                if isclose(ideal_dist_sca[i, j], nn_dist[prox-1], rel_tol=.001):
                    i_neigh[prox].append(j)

    neighbors.append(i_neigh)
#print(neighbors[0])
#sys.exit()

fc_mat = np.zeros((n, n, 3, 3))
base_mat = np.zeros((6, 3, 3))
nn_matrix = .5 * a_val * np.array([[0.,0.,0.],
                                   [1.,1.,1.],
                                   [2.,0.,0.],
                                   [0.,2.,2.],
                                   [3.,1.,1.],
                                   [2.,2.,2.]])
#print(nn_matrix)


fc_arr = np.array([[0, 0, 14, 14],
                   [1, 1, 2, 2],
                   [3, 4, 14, 14],
                   [5, 6, 14, 7],
                   [8, 9, 10, 11],
                   [12, 12, 13, 13]])
#print(fc_arr.shape)
#print(fc_arr)
#sys.exit()
# For B2
#fc_in_path = '/Users/jamunoz/OneDrive - University of Texas at El Paso/FeV_lammps/force_constants/'+root+'_7x7x7/'


[α0, α1, β1, α2, β2] = [ 8.33366357, -0.65260093, -0.20014162, -0.00126518, -0.77779512 ]
α3 = 0#0.0535625 
β3 = 0#-0.0975
γ3 = 0#-0.06625
α4 = 0#-0.0473125
β4 = 0#-0.0011875
γ4 = 0#-0.0015
δ4 = 0#-0.03025
α5 = 0#0.018125
β5 = 0#-0.02
#α0 =  (8*α1 + 2*α2 + 4*β2 + 4*α3 + 8*β3 + 8*α4 + 16*β4 + 8*α5)
#α0 = -(8*α1 + 2*α2 + 4*β2 + 4*α3 + 8*β3 + 8*α4 + 16*β4 + 8*α5)
fc = [α0, α1, β1, α2, β2, α3, β3, γ3, α4, β4, γ4, δ4, α5, β5, 0.0]



for prox in range(6):
    base_mat[prox, :, :] = np.array([[fc[fc_arr[prox, 0]], fc[fc_arr[prox, 2]], fc[fc_arr[prox, 2]]],
                                     [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 1]], fc[fc_arr[prox, 3]]],
                                     [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 3]], fc[fc_arr[prox, 1]]]])

for i in range(250):
    for prox in range(6):
        for j in neighbors[i][prox]:
            fc_mat[i, j, :, :] = base_mat[prox, :, :]
            if prox in [2, 3, 4]:
                for r in range(1,3):
                    if isclose(abs(ideal_distances[i, r, j]), nn_matrix[prox, 0], rel_tol=.001):
                        fc_mat[i, j, [r, 0], :] = fc_mat[i, j, [0, r], :]
                        fc_mat[i, j, :, [r, 0]] = fc_mat[i, j, :, [0, r]]
                        
            for r in range(3):
                if ideal_distances[i, r, j] < 0:
                    fc_mat[i, j, r, :] = -fc_mat[i, j, r, :]
                    fc_mat[i, j, :, r] = -fc_mat[i, j, :, r]
 
np.around(fc_mat, decimals=5)
#print(fc_mat[0:2,0:2])
#sys.exit()
# 0 0 0  -0.5 0.5 0.5  0.25 0.25 0.25 0 0 0 0 0.5 0
out_path = '/Users/amirhusen/Desktop/Amir/Research/Phonopy/'+root+'/'
out_filename = 'FORCE_CONSTANTS'
with open(out_path+out_filename, 'w') as file:
    print(str(n) + ' ' + str(n), file=file)
    for i in range(n):
        for j in range(n):
            print(str(i+1) + ' ' + str(j+1), file=file)
            np.savetxt(file, fc_mat[i, j, :, :])
            
os.chdir(out_path)
command = 'phonopy -p -s band.conf'
os.system(command)

command = 'mv band.yaml band_' + 'ga' + '.yaml'
os.system(command)
command = 'mv band.pdf band_' + 'ga' + '.pdf'
os.system(command)
command = 'rm phonopy.yaml'
os.system(command)

distances = []
                
in_path_y = '/Users/amirhusen/Desktop/Amir/Research/Phonopy/'+root+'/'
in_filename = 'band_' + 'ga' + '.yaml'
#print(in_filename)
with open(in_path_y+in_filename) as file:
    yaml_file = yaml.full_load(file)
phonons = yaml_file['phonon']
phonon_no = len(phonons)
band_contents = []
freqs1 = []
freqs2 = []
freqs3 = []
for j in range(phonon_no):
    band_contents.append(phonons[j]['band'])
    freqs1.append(band_contents[j][0]['frequency'])
    freqs2.append(band_contents[j][1]['frequency'])
    freqs3.append(band_contents[j][2]['frequency'])
    distances.append(phonons[j]['distance'])
#print(distances[:])
#print(freqs[:])

#Ploting our results
plt.plot(distances,freqs1, distances,freqs2, distances,freqs3, color = 'red')

#plt.xlim(0, 2)
plt.ylim(-2, 10)

# Optional: Add labels and legend
#plt.xlabel('q-points', labelpad=17)
#plt.ylabel('Frequency (THz)')
x_label = [distances[0], distances[200], distances[401], distances[602], distances[803]]
y_label = -1
print(x_label)

plt.xticks([])



plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)


# Labels corresponding to each point
labels = [r'$\Gamma$', 'H', 'P', r'$\Gamma$', 'N']

for i, label in enumerate(labels):
    plt.text(x_label[i], y_label, label, fontsize=12, ha='center', va='bottom')
    plt.axvline(x=x_label[i], color='gray', linestyle=':', linewidth=1)


#Data from reference
# Gamma to N
y_G2N_L = np.array([0.30870, 0.45287, 0.57449, 0.67356, 0.75419, 0.79586, 0.86624, 0.92021,
                    0.90039])*7 

x_G2N_L = np.array([0.16089, 0.25055, 0.33748, 0.42451, 0.51162, 0.59466, 0.68181, 0.76621, 
                    0.84809]) * (1.1350355 - 0.9016669) + 0.9016669

y_G2N_T2 = np.array([1.44404, 1.83123, 2.13103, 2.69146, 3.20861, 3.69653, 4.24250, 4.75584,
                     5.45699, 5.52159])

x_G2N_T2 = np.array([0.23245, 0.27813, 0.33230, 0.38232, 0.43089, 0.48372, 0.53373, 0.63493, 
                     0.73906, 0.84577])* (1.1350355 - 0.9016669) + 0.9016669

y_G2N_T1 = np.array([1.69863, 1.99809, 2.39449, 2.91510, 3.32532, 3.69386, 3.99324, 4.08521, 
                     4.31504, 4.47617])

x_G2N_T1 = np.array([0.24408, 0.31462, 0.38388, 0.45740, 0.52667, 0.59866, 0.67058, 0.74092,
                     0.81828, 0.88869]) * (1.1350355 - 0.9016669) + 0.9016669

# H to P to Gamma
x_L = np.array([0.03661, 0.05328, 0.06658, 0.08155, 0.09820, 0.13152, 0.16327, 0.19996, 0.23508, 0.27559, 0.31451, 0.35497, 
                0.39719, 0.43957, 0.48185, 0.52582, 0.57155, 0.61894, 0.66284, 0.71331, 0.76228, 0.81229, 0.86433])
y_L = np.array([1.31375, 1.82568, 2.33750, 2.88346, 3.46357, 4.53857, 5.32375, 6.41591, 7.18415, 7.08326, 6.72663, 6.76210,
                6.61013, 5.99794, 5.69256, 5.35315, 4.82630, 4.36769, 4.21578, 4.52432, 4.35553, 5.90837, 6.52378])

x_T = np.array([0.13036, 0.18661, 0.24439, 0.30509, 0.36423, 0.40854, 0.45428, 0.51468, 0.57656, 0.63993, 0.70640, 0.78010,
                0.84954, 0.89960, 0.94060])
y_T = np.array([2.04632, 2.83213, 3.76634, 4.62629, 5.24879, 5.60435, 5.81146, 5.86993, 5.92838, 6.06102, 6.59436, 6.74169, 7.34919, 7.31877, 6.67994]) 

x_HPG_L = (1 - np.flip(x_L))*(0.9016669 - 0.3300330)+ 0.3300330
y_HPG_L = np.flip(y_L)

x_HPG_T = (1 - np.flip(x_T))*(0.9016669 - 0.3300330)+ 0.3300330
y_HPG_T = np.flip(y_T)


# Gamma to H
x_G2H_L = np.array([0.07288, 0.10077, 0.13488, 0.15812, 0.21549, 0.27131, 0.32555, 0.37984, 0.43583, 0.49018, 0.54294, 0.59432,
                    0.64409, 0.69391, 0.74678, 0.79340, 0.84160, 0.88665])* 0.3496503
y_G2H_L = np.array([1.22399, 1.83611, 2.44814, 2.96610, 3.97044, 4.91197, 6.04201, 6.85791, 6.84132, 7.34309, 8.03336, 7.68700,
                    7.65480, 7.37130, 7.37047, 7.58963, 7.63600, 7.94942])

x_G2H_T = np.array([0.18644, 0.21864, 0.25424, 0.28983, 0.32542, 0.39661, 0.47119, 0.54576, 0.65763, 0.73559, 0.81525, 0.85932, 
                    0.89831])* 0.3496503
y_G2H_T = np.array([1.80000, 2.02286, 2.33143, 2.81143, 3.13714, 4.01143, 4.76571, 5.36571, 6.24000, 7.57714, 7.56000, 7.52571, 
                    7.98857])
# Gamma to H 
plt.scatter(x_G2H_L, y_G2H_L, marker = "o", color ='orange')
plt.scatter(x_G2H_T, y_G2H_T, marker = "P", color = 'green')

# H to P to Gamma
plt.scatter(x_HPG_L, y_HPG_L, marker = "o", color = 'orange')
plt.scatter(x_HPG_T, y_HPG_T, marker = "P", color = 'green')

# Gamma to N
plt.scatter(x_G2N_L, y_G2N_L, marker = "o", color = 'orange')
plt.scatter(x_G2N_T2, y_G2N_T2, marker = "*", color = 'blue')
plt.scatter(x_G2N_T1, y_G2N_T1, marker = "*", color = 'blue')
"""
x_label = [0.00, 0.3300330, 0.6158500, 0.9016669, 1.1350355]
y_label = [9.0] * len(x_label)  # Assuming y-values are zeros or any specific value you want

# Labels corresponding to each point
labels = ['$\Gamma$', 'H', 'P', '$\Gamma$', 'N']

for i, label in enumerate(labels):
    plt.text(x_label[i], y_label[i], label, fontsize=12, ha='right', va='bottom')
"""
plt.show()

sys.exit()




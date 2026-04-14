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
a_val = 2.86 #Side length of each cube in bcc
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



α0 = 9.79702                                                                                                                                                                                                                                                                       
α1 = -0.89286 
β1 = -0.85344 
α2 = -0.54061 
β2 = -0.39364
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


plt.xticks([])



plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)


# Labels corresponding to each point
labels = [r'$\Gamma$', 'H', 'P', r'$\Gamma$', 'N']

for i, label in enumerate(labels):
    plt.text(x_label[i], y_label, label, fontsize=12, ha='center', va='bottom')
    plt.axvline(x=x_label[i], color='gray', linestyle=':', linewidth=1)

#Data from reference

# Gamma to N
y_G2N_L = np.array([7.5, 11.5, 13.8, 16.6, 18.9, 22.3, 25.5, 28.6, 
       31.3, 33.6, 35.5, 36.1, 37.1, 38.2, 38.3, 38.3])/4.1357 #*0.242 

x_G2N_L = np.array([0.141, 0.212, 0.248, 0.301, 0.354, 0.424, 0.495, 0.566, 0.636,
                  0.707, 0.778, 0.813, 0.884, 0.955, 1.025, 1.096])/1.096 * 0.2472402 + 0.9552625

y_G2N_T2 = np.array([7.9, 11.9, 15.4, 18.5, 20.8, 22.4, 23.9, 25.1, 26.1, 26.5, 26.7, 26.7])/4.1357#*0.242

x_G2N_T2 = np.array([0.219, 0.329, 0.438, 0.548, 0.636, 0.707, 0.778, 0.849, 0.919, 0.990, 
                     1.061, 1.096])/1.096 * 0.2472402 + 0.9552625

y_G2N_T1 = np.array([18.5])/4.1357#*0.242

x_G2N_T1 = np.array([1.096])/1.096 * 0.2472402 + 0.9552625

# H to P to Gamma
x_L = np.array([0.231, 0.346, 0.462, 0.577, 0.693, 0.751, 0.808, 0.924, 1.039, 1.097, 1.155, 1.212, 
                1.270, 1.328, 1.386, 1.501, 1.617, 1.730, 1.846, 1.961, 2.077, 2.192])
y_L = np.array([15.5, 22.4, 28.0, 32.7, 33.9, 34.7, 34.6, 33.8, 31.4, 29.8, 28.5, 27.3, 25.9, 24.7, 
                23.8, 23.7, 24.9, 27.7, 30.4, 33.2, 34.9, 35.4]) /4.1357#* 0.242

x_T = np.array([0.231, 0.346, 0.462, 0.577, 0.693, 0.808, 0.924, 1.039, 1.155, 1.270, 1.386, 1.501, 
                1.617, 1.732, 1.848, 1.963, 2.079, 2.192])
y_T = np.array([7.9, 12.0, 16.1, 19.9, 23.4, 25.8, 27.8,  29.4, 30.4, 31.8, 32.9, 33.4, 34.5, 34.4,
                34.9,  35.1, 35.6, 35.4]) /4.1357#* 0.242

x_HPG_L = (2.192 - np.flip(x_L))/2.192*(0.9552625-0.3496503)+0.3496503
y_HPG_L = np.flip(y_L)

x_HPG_T = (2.192 - np.flip(x_T))/2.192*(0.9552625-0.3496503)+0.3496503
y_HPG_T = np.flip(y_T)


# Gamma to H
x_G2H_L = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.192])/2.192 * 0.3496503
y_G2H_L = np.array([13.8, 19.8, 25.0, 29.3, 32.2, 34.2, 35.5, 35.8, 36.0, 35.4])/4.1357#*0.242
x_G2H_T = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.096, 1.315, 1.534, 1.753, 1.973, 2.192])/2.192 * 0.3496503
y_G2H_T = np.array([5.0, 10.2, 15.0, 19.3, 23.3, 25.1, 28.7, 31.4, 33.4, 35.4, 35.4])/4.1357#*0.242

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

x_label = [0.00, 0.3496503, 0.6524564, 0.9552625, 1.2025027]
y_label = [10.0] * len(x_label)  # Assuming y-values are zeros or any specific value you want



plt.show()

sys.exit()




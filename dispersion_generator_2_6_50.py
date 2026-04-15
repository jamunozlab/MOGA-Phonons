# Date: 06/16/2023
# Amir Husen

import numpy as np
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
a_val = 2.2 #Side length of each cube in bcc
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


[α0, α1, β1, α2, β2] = [ 7.99751658e+00, -4.39172161e-01 , 9.86059293e-01 , 2.02414201e-03, -1.12105143e+00]

#[α0, α1, β1, α2, β2] = [ 9.50512721, -0.86342522, -0.24267484,  0.02242717, -0.64877193]
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

out_path = '/Users/jamunoz/Documents/GitHub/MOGA-Phonons/'+root+'/'
#out_path = '/Users/amirhusen/Desktop/Amir/Research/Phonopy/'+root+'/'
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

in_path_y = out_path                
#in_path_y = '/Users/amirhusen/Desktop/Amir/Research/Phonopy/'+root+'/'
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

"""#print(freqs[:])
from matplotlib import pyplot as plt
plt.plot(distances,freqs1, distances,freqs2, distances,freqs3, color = 'red')
#plt.xlim(0, 2)
plt.ylim(-2, 10)

# Optional: Add labels and legend
#plt.xlabel('q-points', labelpad=17)
plt.ylabel('Frequency (THz)')

plt.xticks([])

plt.axvline(x=0, color='gray', linestyle=':', linewidth=1)

plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)

x_label = [distances[0], distances[200], distances[401], distances[602], distances[803]]
y_label = -2.2

# Labels corresponding to each point
labels = ['$\Gamma$', 'H', 'P', '$\Gamma$', 'N']
#points = ['1', '201', '402', '603', '804']

for x, label in zip(x_label, labels):
    plt.text(x, y_label, label, ha='center', va='top', fontsize=10)
    
plt.show()
"""
from matplotlib import pyplot as plt

plt.figure(figsize=(5.5, 4))

# Plot phonon branches
plt.plot(distances, freqs1, color='red', linewidth=1.5)
plt.plot(distances, freqs2, color='red', linewidth=1.5)
plt.plot(distances, freqs3, color='red', linewidth=1.5)

# Y-axis
plt.ylim(-2, 10)
plt.ylabel('Frequency (THz)', fontsize=20)

# High-symmetry points
x_ticks = [distances[0], distances[200], distances[401], distances[602], distances[803]]
x_labels = [r'$\Gamma$', 'H', 'P', r'$\Gamma$', 'N']

plt.xticks(x_ticks, x_labels, fontsize=20)
plt.yticks(fontsize=20)

# Vertical symmetry lines
for x in x_ticks:
    plt.axvline(x=x, color='gray', linestyle=':', linewidth=1)

# Horizontal zero line
plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)

# ✅ KEEP all spines (top/right visible)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)

# ✅ Top-left annotation (two lines)
plt.text(0.05, 0.99, 'm = 50\na = 2.2',
         transform=ax.transAxes,
         fontsize=20,
         verticalalignment='top')

# Layout
plt.tight_layout()

plt.show()
sys.exit()






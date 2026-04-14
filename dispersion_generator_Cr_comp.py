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
a_val = 2.885 #Side length of each cube in bcc
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


[α0, α1, β1, α2, β2] = [ 8.74500349, -0.99806181, -0.57840917,  0.30757177, -0.54716558]
α3 = -0.003121 #0.0535625 
β3 = 0.127526 #-0.0975
γ3 = 0.179243 #-0.06625
α4 = -0.078445 #-0.0473125
β4 = 0.026971 #-0.0011875
γ4 = 0.000437 #-0.0015
δ4 = 0.032208 #-0.03025
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
x_G2N_L = 0.9388491 + np.array([0.077104, 0.129546, 0.215056, 0.274457, 0.349223, 0.417190, 0.485158, 0.557103, 0.633058, 0.701636
])/ 0.702962 * (1.1818411 - 0.9388491)
y_G2N_L = np.array([2.396104, 3.176612, 4.965976, 6.276659, 7.510776, 8.088395, 8.666013, 9.117115, 9.416435, 9.514023
])

x_G2N_T1 = 0.9388491 + np.array([0.205919, 0.220898, 0.232092, 0.258297, 0.273340, 0.292168, 0.314653, 0.333481, 0.344739, 0.363631, 0.374890, 0.389933, 0.401223, 0.420083, 0.450234, 0.488019, 0.522050, 0.559835, 0.593963, 0.631909, 0.666037, 0.704111
])/ 0.702962 * (1.1818411 - 0.9388491)
y_G2N_T1 = np.array([3.147490, 3.374102, 3.575642, 3.978528, 4.154610, 4.355764, 4.683049, 4.884203, 5.035213, 5.185838, 5.336848, 5.512930, 5.638675, 5.814564, 6.116199, 6.417447, 6.668360, 6.969608, 7.144726, 7.319651, 7.494769, 7.568634
])

x_G2N_T2 = 0.9388491 + np.array([0.138852, 0.153863, 0.168906, 0.180100, 0.195143, 0.206369, 0.225197, 0.236423, 0.251467, 0.266510, 0.277704, 0.292747, 0.307822, 0.319080, 0.330339, 0.345414, 0.360489, 0.371780, 0.386887, 0.398145, 0.420919, 0.451198, 0.485294, 0.523272, 0.561217, 0.595506, 0.629795, 0.645417, 0.660878, 0.680028, 0.702962
])/ 0.702962 * (1.1818411 - 0.9388491)
y_G2N_T2 = np.array([1.862458, 2.063805, 2.239887, 2.441427, 2.617509, 2.793783, 2.994937, 3.171212, 3.347294, 3.523376, 3.724916, 3.900998, 4.051815, 4.202825, 4.353836, 4.504653, 4.655470, 4.781215, 4.906768, 5.057778, 5.157680, 5.358255, 5.558638, 5.708298, 5.883223, 5.932017, 5.980810, 5.702126, 5.549766, 5.498272, 5.471850
])


# P to G
x_P2G_L =  0.3436426 + 0.9388491 * (np.array([1.915631, 1.957812, 2.045636, 2.133074, 2.174836, 2.220352, 2.262050, 2.307661, 2.349424, 2.394650, 2.480963, 2.566537, 2.611281
])-1)/2.75
y_P2G_L = np.array([8.686263, 8.839974, 8.869674, 8.596197, 8.421465, 8.196397, 7.971135, 7.821861, 7.647129, 7.194677, 6.036933, 4.298099, 3.466675
])

x_P2G_T = 0.3436426 + 0.9388491 * (np.array([1.888430, 1.937955, 2.002845, 2.025715, 2.063692, 2.090347, 2.109272, 2.132013, 2.154723, 2.177336, 2.200013, 2.222690, 2.237765, 2.256593, 2.275454, 2.290529, 2.309325, 2.324368, 2.343164, 2.361960, 2.380788, 2.395735, 2.414530, 2.429509, 2.444488, 2.463252, 2.478199, 2.496995, 2.511909, 2.530641, 2.545620
])-1)/2.75
y_P2G_T = np.array([8.305942, 8.232655, 8.235933, 8.211826, 8.062166, 8.012986, 7.887626, 7.762460, 7.612028, 7.385802, 7.210106, 7.034410, 6.883592, 6.682438, 6.506549, 6.355732, 6.129313, 5.953231, 5.726812, 5.500394, 5.299240, 5.047363, 4.820945, 4.594333, 4.367722, 4.116038, 3.864162, 3.637743, 3.360602, 3.083653, 2.857042
]) 

# H to P
x_H2P_L = 0.3436426 + 0.6412458 * (np.array([1.044189, 1.090090, 1.132174, 1.174161, 1.261986, 1.349906, 1.433881, 1.521737, 1.609336, 1.697096, 1.784888, 1.827293, 1.869634
])-1)/1.869634
y_H2P_L = np.array([7.732759, 7.810868, 7.888784, 7.890905, 7.920606, 8.026101, 8.030343, 8.085309, 7.938156, 7.917327, 7.921763, 8.252326, 8.532360
])

x_H2P_T = 0.3436426 + 0.6412458 * (np.array([1.044125, 1.086016, 1.131981, 1.262178, 1.350260, 1.438180, 1.522251, 1.610075, 1.698092, 1.785820, 1.827711
])-1)/1.869634
y_H2P_T = np.array([7.682229, 7.608557, 7.737195, 8.072194, 8.304013, 8.409508, 8.489545, 8.519246, 8.700535, 8.654441, 8.580769
]) 



# Gamma to H
x_G2H_L = np.array([0.090974, 0.185122, 0.283088, 0.380410, 0.458037, 0.599814, 0.702970, 0.748582, 0.816838, 0.850420, 0.876915, 0.899463, 0.925957, 0.948763, 1.002202
])* 0.3436426
y_G2H_L = np.array([2.505649, 4.506196, 6.506935, 8.002379, 9.016827, 9.453464, 9.534466, 9.385191, 9.034956, 8.430337, 8.254834, 7.978078, 7.802575, 7.727938, 7.730638
])

x_G2H_T = np.array([0.155124, 0.178477, 0.201797, 0.228934, 0.252254, 0.275606, 0.302743, 0.325999, 0.349287, 0.372543, 0.399648, 0.426657, 0.449848, 0.476825, 0.499920, 0.549991, 0.596181, 0.649973, 0.699884, 0.749924, 0.799770, 0.826522, 0.872262, 0.898981, 0.921883, 0.948570, 1.001977, 0.971473, 0.857219
])* 0.3436426
y_G2H_T = np.array([1.927838, 2.282702, 2.612301, 2.942094, 3.271693, 3.626557, 3.956349, 4.235419, 4.539753, 4.818823, 5.123351, 5.352084, 5.580624, 5.784092, 5.936838, 6.293052, 6.598544, 6.879157, 7.109047, 7.439996, 7.619357, 7.645972, 7.597756, 7.599106, 7.600264, 7.576349, 7.553784, 7.577506, 7.773838
])

# Gamma to H 
plt.scatter(x_G2H_L, y_G2H_L, marker = "o", color ='orange')
plt.scatter(x_G2H_T, y_G2H_T, marker = "P", color = 'green')


# H to P 
plt.scatter(x_H2P_L, y_H2P_L, marker = "o", color = 'orange')
plt.scatter(x_H2P_T, y_H2P_T, marker = "P", color = 'green')


# P to G 
plt.scatter(x_P2G_L, y_P2G_L, marker = "o", color = 'orange')
plt.scatter(x_P2G_T, y_P2G_T, marker = "P", color = 'green')

# Gamma to N
plt.scatter(x_G2N_L, y_G2N_L, marker = "o", color = 'orange')
plt.scatter(x_G2N_T2, y_G2N_T2, marker = "*", color = 'blue')
plt.scatter(x_G2N_T1, y_G2N_T1, marker = "*", color = 'blue')



plt.show()

sys.exit()




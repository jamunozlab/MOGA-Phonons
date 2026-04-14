#!/usr/bin/env python
# coding: utf-8

# ### Import required libraries

# In[2]:


import numpy as np
from math import sqrt, isclose
#import h5py
import os, sys, yaml, time
import pandas as pd
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from geneticalgorithm import geneticalgorithm as ga


# In[3]:


t1 = time.time()
# parameter setting
root = 'B2_2.86_10K'
n = 250 # Number of atoms
ndisps = 5 # Number of dispersions to generate (equal to number of yalm files)
a_val = 2.86
system_size = 5
alat = a_val * system_size # the size of the box
total_atoms = 2 * (system_size ** 3)
init_no = 0


# Defining unit cell

unit_cell_length = a_val #lattice_parameter#*2

basis = np.eye(3)*unit_cell_length

base_atoms = np.array([[0.0, 0.0, 0.0],
                       [0.5, 0.5, 0.5]])*unit_cell_length

# Generate atom positions
positions = []
for i in range(system_size): 
    for j in range(system_size):
        for k in range(system_size):
            base_position = np.array([i, j, k]) #Local variable
            cart_position = np.inner(basis.T, base_position) #Local
            for atom in base_atoms:
                positions.append(cart_position + atom)
ide_lat = np.array(positions)

#a_val = alat / ((n / 2.)**(1./3.)) #Side length of each cube in bcc

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

fc_mat = np.zeros((n, n, 3, 3))
base_mat = np.zeros((6, 3, 3))
nn_matrix = .5 * a_val * np.array([[0.,0.,0.],
                                   [1.,1.,1.],
                                   [2.,0.,0.],
                                   [0.,2.,2.],
                                   [3.,1.,1.],
                                   [2.,2.,2.]])

# print(nn_matrix)
# sys.exit()
fc_arr = np.array([[0, 0, 14, 14],# alfa, beta, gama, delta
                   [1, 1, 2, 2],
                   [3, 4, 14, 14],
                   [5, 6, 14, 7],
                   [8, 9, 10, 11],
                   [12, 12, 13, 13]])


# In[4]:


def f(X):
    # α0 + 8α1 + 2α2 + 4β2 + 4α3 + 8β3 + 8α4 + 16β4 + 8α5 = 0 and α1 = X[0] = [-0.85, -0.55]
    alpha0 = X[0]
    alpha1 = X[1]
    beta1 = X[2]# -0.9375
    alpha2 = X[3]# -0.650625
    beta2 = X[4]# -0.245625 
    alpha3 = 0.0535625 
    beta3 = -0.0975
    gamma3 = -0.06625
    alpha4 = -0.0473125
    beta4 = -0.0011875
    gamma4 = -0.0015
    delta4 = -0.03025
    alpha5 = 0.018125
    beta5 = -0.02
    #α0 = -(8*α1 + 2*α2 + 4*β2 + 4*α3 + 8*β3 + 8*α4 + 16*β4 + 8*α5)
    fc = [alpha0, alpha1, beta1, alpha2, beta2, alpha3, beta3, gamma3, alpha4, beta4, gamma4, delta4, alpha5, beta5, 0.0]
    
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
                            fc_mat[i, j, :, [r, 0]] = 0
                            
                for r in range(3):
                    if ideal_distances[i, r, j] < 0:
                        fc_mat[i, j, r, :] = -fc_mat[i, j, r, :]
                        fc_mat[i, j, :, r] = -fc_mat[i, j, :, r]
                        
    np.around(fc_mat, decimals=5)
    
    out_path = '/Users/amirhusen/Desktop/Amir/Research/phonopy/'+root+'/'
    out_filename = 'FORCE_CONSTANTS'
    with open(out_path+out_filename, 'w') as file:
        print(str(n) + ' ' + str(n), file=file)
        for i in range(n):
            for j in range(n):
                print(str(i+1) + ' ' + str(j+1), file=file)
                np.savetxt(file, fc_mat[i, j, :, :])
    
    in_path = out_path

    os.chdir(out_path)
    command = 'phonopy -p -s band.conf'
    os.system(command)
    
    command = 'mv band.yaml band_' + 'ga' + '.yaml'
    os.system(command)
    command = 'mv band.pdf band_' + 'ga' + '.pdf'
    os.system(command)
    command = 'rm phonopy.yaml'
    os.system(command)
    
                        
 
    in_filename = 'band_' + 'ga' + '.yaml'
    #print(in_filename)
    with open(in_path+in_filename) as file:
        yaml_file = yaml.full_load(file)
    phonons = yaml_file['phonon']
    phonon_no = len(phonons)
    band_contents = []
    freqs = []
    for j in range(phonon_no):
        band_contents.append(phonons[j]['band'])
        freqs.append(band_contents[j][0]['frequency'])
    # Taking negative frequencies
    neg_freqs = []
    for i in range(len(freqs)):
        if freqs[i] < 0:
            neg_freqs.append(freqs[i])
    """
    squared_freqs = [number**2 for number in freqs] # for all frequency
    rmse = sqrt(sum(squared_freqs)/len(freqs))
    """
    pen1 = 0
    if len(neg_freqs) != 0:
        pen1 =len(neg_freqs)
    
            
    pen2 = 0
    total = (alpha0 + 8*alpha1 + 2*alpha2 + 4*beta2 + 4*alpha3 + 8*beta3 + 8*alpha4 + 16*beta4 + 8*alpha5)
    if total != 0:
        pen2 = abs(total)
    if freqs[0] != 0:
        pen3 = 1000*abs(freqs[0])
    return (pen1+pen2+pen3)


# In[5]:


varbound=np.array([[1, 10],[-1.00, 1.00], [-1.00, 1.00], [-1.00, 1.00], [-1.00, 1.00]])

algorithm_param = {'max_num_iteration': 200,\
                   'population_size':20,\
                   'mutation_probability':0.10,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=f,\
            dimension=5,\
            variable_type='real',\
            function_timeout=60,\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)


model.run()
t2 = time.time()
print("Time is", t2-t1, "Sec")


# In[ ]:





# In[ ]:





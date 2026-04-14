# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:30:42 2023

@author: ahusen
"""
#import seaborn as sb
#import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, isclose
#import h5py
import os, sys, yaml
import pandas as pd
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
#from geneticalgorithm import geneticalgorithm as ga
import pygad
import time

t1 = time.time()
# parameter setting
root = '2_6_50'
n = 250 # Number of atoms
#ndisps = 5 # Number of dispersions to generate (equal to number of yalm files)
a_val = 2.40
system_size = 5
alat = a_val * system_size # the size of the box



# Defining unit cell

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


def fitness_func(ga_instance, solution, solution_idx):
    # α0 + 8α1 + 2α2 + 4β2 + 4α3 + 8β3 + 8α4 + 16β4 + 8α5 = 0 and α1 = X[0] = [-0.85, -0.55]
    α0, α1, β1, α2, β2  = solution 
    #β1 = -0.9375
    #α2 = 0 #-0.650625
    #β2 = 0 # -0.245625 
    α3 = 0 #0.0535625 
    β3 = 0 #-0.0975
    γ3 = 0 #-0.06625
    α4 = 0 #-0.0473125
    β4 = 0 #-0.0011875
    γ4 = 0 #-0.0015
    δ4 = 0#-0.03025
    α5 = 0 #0.018125
    β5 = 0 #-0.02

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
   
    
    
    """
    path = [[[0.0, 0.0, 0.0], [-0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]]
    labels = ["$\Gamma$", "H", "P",  "$\Gamma$", "N"]
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=201)
    atomic_masses = [55.85]
    
    phonons = phonopy.load(supercell_matrix=[5, 5, 5],
                  primitive_matrix ='auto',
                  unitcell_filename= out_path+"POSCAR",
                  force_constants_filename= out_path+out_filename)
    phonons.set_masses(atomic_masses)
    
    #phonons._force_constants_decimals = 5
    #sys.exit()
    
    phonons.run_band_structure(qpoints, path_connections=connections, labels=labels)
    #phonons.plot_band_structure().show()
    #sys.exit()
    res=phonons.get_band_structure_dict()['frequencies']
    #print(len(res))
    
    freqs=[]
    for i in range(4):
        for j in range(201):
            freqs.append(res[i][j][0])  
    """
   
    os.chdir(out_path)
    command = 'phonopy -p -s band.conf'
    os.system(command)
    
    command = 'mv band.yaml band_' + 'ga' + '.yaml'
    os.system(command)
    command = 'mv band.pdf band_' + 'ga' + '.pdf'
    os.system(command)
    command = 'rm phonopy.yaml'
    os.system(command)
    
                        
    in_path_y = '/Users/amirhusen/Desktop/Amir/Research/Phonopy/'+root+'/'
    in_filename = 'band_' + 'ga' + '.yaml'
    #print(in_filename)
    with open(in_path_y+in_filename) as file:
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
    
    #squared_freqs = [number**2 for number in freqs] # for all
    #rmse = sqrt(sum(squared_freqs)/len(freqs))
    #print(len(neg_freqs))
    #sys.exit()
    
    fitness1 = 1.0/(1 + len(neg_freqs))
    """
    neg_squared_freqs = [number**2 for number in neg_freqs] # only for negative
    if len(neg_freqs) == 0:
        rmse = 0 
    else:
        rmse = sqrt(sum(neg_squared_freqs)/len(neg_freqs))
    
    fitness1 = 1.0/(1 + rmse)

    """
    
    total = abs(α0 + 8*α1 + 2*α2 + 4*β2 + 4*α3 + 8*β3 + 8*α4 + 16*β4 + 8*α5)
    fitness2= 1.0/(1 + total)
    
    fitness3 = 1.0/(1 + abs(freqs[0]))
   
    #print(fitness1 ,fitness2, fitness3)
    #sys.exit()
    

    return [fitness1, fitness2, fitness3]
    

#print(f([-0.1]))
def on_generation(ga_instance):
    # Retrieve the best solution in the current generation
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    with open("generation_output.txt", "a") as file:
        file.write(f"Generation {ga_instance.generations_completed}:\n")
        file.write(f"    Best solution: {best_solution}\n")
        file.write(f"    Fitness value: {best_solution_fitness}\n")
        file.write(f"    Index: {best_solution_idx}\n")
        file.write("\n")  # A blank line for readability
        
    #print(f"Generation {ga_instance.generations_completed}:")
    #print(f"    Best solution: {best_solution}")
    #print(f"    Fitness value: {best_solution_fitness}")
    #print(f"    Index: {best_solution_idx}")
    #time.sleep(1)  # Pause for 1 seconds


if __name__ == '__main__':
    # Genetic Algorithm parameters
    num_generations = 300
    sol_per_pop = 20
    num_parents_mating = 12
    num_genes = 5
    
    # Define the bounds for alpha0 and alpha1
    gene_space = [{'low': 1, 'high': 10}, {'low': -2.00, 'high': 2.00}, {'low': -2.00, 'high': 2.00}, {'low': -2.00, 'high': 2.00}, {'low': -2.00, 'high': 2.00}]
    
    # Preserve last population
    
    
    # Initialize the genetic algorithm
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           mutation_percent_genes=10,
                           #mutation_num_genes=1,
                           parent_selection_type = "nsga2",
                           keep_elitism= 2,
                           crossover_type = "uniform",
                           mutation_type = "random",
                           #gene_type=float,
                          
                           #delay_after_gen = 1,
                           #parallel_processing=8,
                           #save_best_solutions=True,
                           on_generation = on_generation,
                           #stop_criteria=["reach_0.99999", "saturate_20"]
                           #initial_population=last_population
                           )

    # Run the genetic algorithm
    
    
    ga_instance.run()

    
    # Retrieve the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution: {solution}")
    print(f"Fitness of the best solution: {solution_fitness}")
    
    
    # Plot the evolution
    ga_instance.plot_fitness(label=['Fitness1', 'Fitness2', 'Fitness3'])
    # Plot the gene values
    #ga_instance.plot_genes(graph_type="plot")  # Use "plot", "histogram", or "box" as needed
    t2 = time.time()
    print("Time is", t2-t1)
    
  
   
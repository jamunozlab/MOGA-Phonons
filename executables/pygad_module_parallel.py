# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:30:42 2023

@author: ahusen
"""

import numpy as np
from math import sqrt, isclose
import os
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms
import pygad
import time
import multiprocessing as mp
import re
import json
import socket
import platform
from datetime import datetime

t1 = time.time()


def read_crystal_params(filename="inputc"):
    """
    Reads crystal parameters from a simple text file.

    Expected format:
        atomic_masses = 80.0
        a_val = 2.20

    atomic_masses may be comma-separated:
        atomic_masses = 80.0, 80.0
    """

    params = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            params[key] = value

    if "a_val" not in params:
        raise ValueError("crystal_params.txt must define a_val")

    if "atomic_masses" not in params:
        raise ValueError("crystal_params.txt must define atomic_masses")

    a_val = float(params["a_val"])

    atomic_masses = [
        float(x) for x in re.split(r"[,\s]+", params["atomic_masses"].strip())
        if x
    ]

    return atomic_masses, a_val

# ============================================================================
# Parameter setting
# ============================================================================
root = '2_6_50'
n = 250
atomic_masses, a_val = read_crystal_params("inputc")
system_size = 5
alat = a_val * system_size

out_path = '/Users/jamunoz/Documents/GitHub/MOGA-Phonons/' + root + '/'

# ============================================================================
# Define unit cell / supercell positions
# ============================================================================
positions = []

basis = np.eye(3) * a_val

base_atoms = np.array([[0.0, 0.0, 0.0]]) * a_val
for i in range(system_size):
    for j in range(system_size):
        for k in range(system_size):
            base_position = np.array([k, j, i])
            cart_position = np.inner(basis.T, base_position)
            for atom in base_atoms:
                positions.append(cart_position + atom)

base_atoms = np.array([[0.5, 0.5, 0.5]]) * a_val
for i in range(system_size):
    for j in range(system_size):
        for k in range(system_size):
            base_position = np.array([k, j, i])
            cart_position = np.inner(basis.T, base_position)
            for atom in base_atoms:
                positions.append(cart_position + atom)

ide_lat = np.array(positions)

# ============================================================================
# Compute ideal distances with MIC
# ============================================================================
ideal_distances = np.zeros((n, 3, n))
ideal_dist_sca = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            ideal_distances[i, :, j] = ide_lat[j, :] - ide_lat[i, :]

            for d in range(3):
                if ideal_distances[i, d, j] > (alat / 2):
                    ideal_distances[i, d, j] -= alat
                elif ideal_distances[i, d, j] <= (-alat / 2):
                    ideal_distances[i, d, j] += alat

            ideal_dist_sca[i, j] = np.linalg.norm(ideal_distances[i, :, j])

# ============================================================================
# Neighbor lists
# ============================================================================
neighbors = []

first = (sqrt(3) * a_val) / 2
second = a_val
third = sqrt(2) * a_val
fourth = (sqrt(11) * a_val) / 2
fifth = sqrt(3) * a_val
nn_dist = [first, second, third, fourth, fifth]

for i in range(n):
    i_neigh = [[i]]
    for prox in range(5):
        i_neigh.append([])

    for j in range(n):
        if i != j:
            for prox in range(1, 6):
                if isclose(ideal_dist_sca[i, j], nn_dist[prox - 1], rel_tol=.001):
                    i_neigh[prox].append(j)

    neighbors.append(i_neigh)

# ============================================================================
# FC bookkeeping
# ============================================================================
nn_matrix = .5 * a_val * np.array([
    [0., 0., 0.],
    [1., 1., 1.],
    [2., 0., 0.],
    [0., 2., 2.],
    [3., 1., 1.],
    [2., 2., 2.]
])

fc_arr = np.array([
    [0, 0, 14, 14],
    [1, 1, 2, 2],
    [3, 4, 14, 14],
    [5, 6, 14, 7],
    [8, 9, 10, 11],
    [12, 12, 13, 13]
])

# ============================================================================
# Band path
# ============================================================================
path = [[[0.0, 0.0, 0.0],
         [-0.5, 0.5, 0.5],
         [0.25, 0.25, 0.25],
         [0.0, 0.0, 0.0],
         [0.0, 0.5, 0.0]]]

labels = [r"$\Gamma$", "H", "P", r"$\Gamma$", "N"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=201)

# ============================================================================
# Per-process phonopy cache
# ============================================================================
_PHONONS = None

def make_bcc_unitcell(a_val, atomic_masses, symbol="Fe"):
    """
    Build the two-atom conventional BCC unit cell directly in memory.

    This replaces the need for a POSCAR file. The generated cell is equivalent to:

        scale = a_val
        lattice = identity matrix
        scaled positions = (0, 0, 0) and (1/2, 1/2, 1/2)

    Parameters
    ----------
    a_val : float
        Cubic lattice parameter in Angstrom.
    atomic_masses : list[float]
        Atomic masses in amu. If one mass is provided, it is used for both BCC atoms.
        If two masses are provided, they are assigned to the two basis atoms.
    symbol : str
        Chemical symbol used by Phonopy. The mass is overwritten by atomic_masses.
    """
    masses = list(atomic_masses)
    if len(masses) == 1:
        masses = [masses[0], masses[0]]
    elif len(masses) != 2:
        raise ValueError(
            "For the two-atom BCC unit cell, atomic_masses must contain either "
            "one value or two values."
        )

    return PhonopyAtoms(
        symbols=[symbol, symbol],
        cell=np.eye(3) * a_val,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        masses=masses,
    )


def get_phonons():
    """
    Create one Phonopy object per worker process and reuse it.
    This avoids sharing mutable phonopy state across processes.
    """
    global _PHONONS
    if _PHONONS is None:
        unitcell = make_bcc_unitcell(a_val, atomic_masses)
        ph = phonopy.Phonopy(
            unitcell,
            supercell_matrix=[system_size, system_size, system_size],
            primitive_matrix="auto",
        )

        _PHONONS = ph

    return _PHONONS

# ============================================================================
# Fitness function
# ============================================================================
def fitness_func(ga_instance, solution, solution_idx):
    α0, α1, β1, α2, β2 = solution

    α3 = 0.0
    β3 = 0.0
    γ3 = 0.0
    α4 = 0.0
    β4 = 0.0
    γ4 = 0.0
    δ4 = 0.0
    α5 = 0.0
    β5 = 0.0

    fc = [α0, α1, β1, α2, β2, α3, β3, γ3, α4, β4, γ4, δ4, α5, β5, 0.0]

    fc_mat = np.zeros((n, n, 3, 3))
    base_mat = np.zeros((6, 3, 3))

    for prox in range(6):
        base_mat[prox, :, :] = np.array([
            [fc[fc_arr[prox, 0]], fc[fc_arr[prox, 2]], fc[fc_arr[prox, 2]]],
            [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 1]], fc[fc_arr[prox, 3]]],
            [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 3]], fc[fc_arr[prox, 1]]]
        ])

    for i in range(n):
        for prox in range(6):
            for j in neighbors[i][prox]:
                fc_mat[i, j, :, :] = base_mat[prox, :, :]

                if prox in [2, 3, 4]:
                    for r in range(1, 3):
                        if isclose(abs(ideal_distances[i, r, j]), nn_matrix[prox, 0], rel_tol=.001):
                            fc_mat[i, j, [r, 0], :] = fc_mat[i, j, [0, r], :]
                            fc_mat[i, j, :, [r, 0]] = fc_mat[i, j, :, [0, r]]
                            fc_mat[i, j, :, [r, 0]] = 0

                for r in range(3):
                    if ideal_distances[i, r, j] < 0:
                        fc_mat[i, j, r, :] = -fc_mat[i, j, r, :]
                        fc_mat[i, j, :, r] = -fc_mat[i, j, :, r]

    fc_mat = np.round(fc_mat, 5)

    phonons = get_phonons()
    phonons.force_constants = fc_mat
    phonons.run_band_structure(qpoints, path_connections=connections, labels=labels)
    res = phonons.get_band_structure_dict()['frequencies']

    freqs = []
    for i in range(len(res)):
        for j in range(len(res[i])):
            freqs.append(res[i][j][0])

    neg_freqs = [f for f in freqs if f < 0]

    fitness1 = 1.0 / (1 + len(neg_freqs))

    total = abs(α0 + 8 * α1 + 2 * α2 + 4 * β2 + 4 * α3 + 8 * β3 + 8 * α4 + 16 * β4 + 8 * α5)
    fitness2 = 1.0 / (1 + total)

    fitness3 = 1.0 / (1 + abs(freqs[0]))

    return [fitness1, fitness2, fitness3]

# ============================================================================
# Generation callback
# ============================================================================
def on_generation(ga_instance):

    generation = ga_instance.generations_completed

    population = ga_instance.population
    fitnesses = ga_instance.last_generation_fitness

    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution(
        pop_fitness=fitnesses
    )

    generation_data = {

        "generation": generation,

        "timestamp": datetime.now().isoformat(),

        "population": np.array(population).tolist(),

        "fitnesses": np.array(fitnesses).tolist(),

        "best_solution_index": int(best_solution_idx),

        "best_solution": np.array(best_solution).tolist(),

        "best_fitness": np.array(best_solution_fitness).tolist(),

        "best_fitness_norm": float(
            np.linalg.norm(best_solution_fitness)
        ),
    }

    with open("generation_history.jsonl", "a") as file:
        file.write(json.dumps(generation_data) + "\n")

    t2 = time.time()

    print(
        f"Generation {generation} | "
        f"Best fitness norm = "
        f"{np.linalg.norm(best_solution_fitness):.6f} | "
        f"Elapsed = {t2 - t1:.2f} s"
    )

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f"Using {nproc} worker processes")

    num_generations = 500
    sol_per_pop = 200
    num_parents_mating = 80
    num_genes = 5

    gene_space = [
    {"low": 3.0,  "high": 10.0},   # alpha0
    {"low": -1.5, "high": 0.4},    # alpha1
    {"low": -1.6, "high": 1.6},    # beta1
    {"low": -2.0, "high": 1.2},    # alpha2
    {"low": -2.0, "high": 0.4},    # beta2
]

    print('reached here')

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space,
        #mutation_percent_genes=20,
        mutation_probability=0.10,
        mutation_num_genes=1,
        parent_selection_type="nsga2",
        keep_elitism=20,
        crossover_type="uniform",
        mutation_type="random",
        on_generation=on_generation,
        parallel_processing=["process", nproc],  # adjust 8 to your machine
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution: {solution}")
    print(f"Fitness of the best solution: {solution_fitness}")

    #ga_instance.plot_fitness(label=['Fitness1', 'Fitness2', 'Fitness3'])

    t2 = time.time()
    print("Time is", t2 - t1)
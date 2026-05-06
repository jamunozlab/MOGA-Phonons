#!/usr/bin/env python3

import os
import re
import argparse
import subprocess
import numpy as np
import yaml
from math import sqrt, isclose


def read_inputc(input_path):
    params = {}

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                key, value = parts

            params[key.strip()] = value.strip()

    a_val = float(params["a_val"])

    atomic_masses = np.array([
        float(x)
        for x in re.split(r"[,\s]+", params["atomic_masses"])
        if x
    ])

    return a_val, atomic_masses


def read_solution(solution_file):
    data = np.ravel(np.loadtxt(solution_file))

    if len(data) != 5:
        raise ValueError(
            f"Expected 5 BvK parameters: alpha0 alpha1 beta1 alpha2 beta2. "
            f"Got {len(data)} values."
        )

    return data


def count_atoms_in_poscar(poscar_path):
    with open(poscar_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # VASP 5 format:
    # line 5: element symbols
    # line 6: atom counts
    # zero-indexed: lines[5]
    try:
        counts = [int(x) for x in lines[6].split()]
        return sum(counts)
    except (ValueError, IndexError):
        pass

    # VASP 4 format:
    # line 5: atom counts
    # zero-indexed: lines[5]
    try:
        counts = [int(x) for x in lines[5].split()]
        return sum(counts)
    except (ValueError, IndexError):
        pass

    raise ValueError(f"Could not determine atom count from POSCAR: {poscar_path}")


def build_bcc_force_constants(a_val, solution, system_size=5):
    alpha0, alpha1, beta1, alpha2, beta2 = solution

    n = 2 * system_size**3
    alat = a_val * system_size

    positions = []
    basis = np.eye(3) * a_val

    for atom_basis in [np.array([[0.0, 0.0, 0.0]]), np.array([[0.5, 0.5, 0.5]])]:
        atom_basis = atom_basis * a_val

        for i in range(system_size):
            for j in range(system_size):
                for k in range(system_size):
                    base_position = np.array([k, j, i])
                    cart_position = np.inner(basis.T, base_position)

                    for atom in atom_basis:
                        positions.append(cart_position + atom)

    ideal_lattice = np.array(positions)

    ideal_distances = np.zeros((n, 3, n))
    ideal_dist_sca = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            ideal_distances[i, :, j] = ideal_lattice[j, :] - ideal_lattice[i, :]

            for d in range(3):
                if ideal_distances[i, d, j] > alat / 2:
                    ideal_distances[i, d, j] -= alat
                elif ideal_distances[i, d, j] <= -alat / 2:
                    ideal_distances[i, d, j] += alat

            ideal_dist_sca[i, j] = np.linalg.norm(ideal_distances[i, :, j])

    nn_dist = [
        sqrt(3) * a_val / 2,
        a_val,
        sqrt(2) * a_val,
        sqrt(11) * a_val / 2,
        sqrt(3) * a_val,
    ]

    neighbors = []

    for i in range(n):
        i_neigh = [[i]]

        for _ in range(5):
            i_neigh.append([])

        for j in range(n):
            if i == j:
                continue

            for prox in range(1, 6):
                if isclose(ideal_dist_sca[i, j], nn_dist[prox - 1], rel_tol=0.001):
                    i_neigh[prox].append(j)

        neighbors.append(i_neigh)

    fc_mat = np.zeros((n, n, 3, 3))
    base_mat = np.zeros((6, 3, 3))

    nn_matrix = 0.5 * a_val * np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 2.0],
            [3.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )

    fc_arr = np.array(
        [
            [0, 0, 14, 14],
            [1, 1, 2, 2],
            [3, 4, 14, 14],
            [5, 6, 14, 7],
            [8, 9, 10, 11],
            [12, 12, 13, 13],
        ]
    )

    alpha3 = beta3 = gamma3 = 0.0
    alpha4 = beta4 = gamma4 = delta4 = 0.0
    alpha5 = beta5 = 0.0

    fc = [
        alpha0,
        alpha1,
        beta1,
        alpha2,
        beta2,
        alpha3,
        beta3,
        gamma3,
        alpha4,
        beta4,
        gamma4,
        delta4,
        alpha5,
        beta5,
        0.0,
    ]

    for prox in range(6):
        base_mat[prox, :, :] = np.array(
            [
                [fc[fc_arr[prox, 0]], fc[fc_arr[prox, 2]], fc[fc_arr[prox, 2]]],
                [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 1]], fc[fc_arr[prox, 3]]],
                [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 3]], fc[fc_arr[prox, 1]]],
            ]
        )

    for i in range(n):
        for prox in range(6):
            for j in neighbors[i][prox]:
                fc_mat[i, j, :, :] = base_mat[prox, :, :]

                if prox in [2, 3, 4]:
                    for r in range(1, 3):
                        if isclose(
                            abs(ideal_distances[i, r, j]),
                            nn_matrix[prox, 0],
                            rel_tol=0.001,
                        ):
                            fc_mat[i, j, [r, 0], :] = fc_mat[i, j, [0, r], :]
                            fc_mat[i, j, :, [r, 0]] = fc_mat[i, j, :, [0, r]]

                for r in range(3):
                    if ideal_distances[i, r, j] < 0:
                        fc_mat[i, j, r, :] = -fc_mat[i, j, r, :]
                        fc_mat[i, j, :, r] = -fc_mat[i, j, :, r]

    return np.around(fc_mat, decimals=5)


def write_force_constants(fc_mat, output_path):
    n = fc_mat.shape[0]

    with open(output_path, "w") as f:
        print(f"{n} {n}", file=f)

        for i in range(n):
            for j in range(n):
                print(f"{i + 1} {j + 1}", file=f)
                np.savetxt(f, fc_mat[i, j, :, :])


def write_band_conf(output_path, atomic_masses, poscar_path):
    n_atoms = count_atoms_in_poscar(poscar_path)

    if len(atomic_masses) == 1:
        masses = [atomic_masses[0]] * n_atoms
    elif len(atomic_masses) == n_atoms:
        masses = atomic_masses
    else:
        raise ValueError(
            f"Mass mismatch: POSCAR has {n_atoms} atoms, "
            f"but inputc has {len(atomic_masses)} masses."
        )

    mass_string = " ".join(str(x) for x in masses)

    contents = f"""ATOM_NAME = X
MASS = {mass_string}

DIM = 1 1 1

BAND = 0.0 0.0 0.0  0.5 -0.5 0.5,  0.5 -0.5 0.5  0.25 0.25 0.25,  0.25 0.25 0.25  0.0 0.0 0.0,  0.0 0.0 0.0  0.0 0.0 0.5
BAND_POINTS = 201
BAND_LABELS = $\\Gamma$ H P $\\Gamma$ N
FORCE_CONSTANTS = READ
"""

    with open(output_path, "w") as f:
        f.write(contents)


def read_band_yaml(band_yaml_path):
    with open(band_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    q_distance = []
    frequencies = []

    for qpoint in data["phonon"]:
        q_distance.append(qpoint["distance"])
        frequencies.append([band["frequency"] for band in qpoint["band"]])

    return np.array(q_distance), np.array(frequencies)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phonopy dispersion data from one MOGA BvK solution."
    )

    parser.add_argument("--simulation_path", required=True)
    parser.add_argument("--solution_file", required=True)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--system_size", type=int, default=5)
    parser.add_argument("--phonopy_command", default="phonopy")

    args = parser.parse_args()

    simulation_path = os.path.abspath(args.simulation_path)

    input_path = os.path.join(simulation_path, "inputc")
    poscar_path = os.path.join(simulation_path, "POSCAR")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Missing inputc: {input_path}")

    if not os.path.isfile(poscar_path):
        raise FileNotFoundError(f"Missing POSCAR: {poscar_path}")

    a_val, atomic_masses = read_inputc(input_path)
    solution = read_solution(args.solution_file)

    fc_mat = build_bcc_force_constants(
        a_val=a_val,
        solution=solution,
        system_size=args.system_size,
    )

    force_constants_path = os.path.join(simulation_path, "FORCE_CONSTANTS")
    band_conf_path = os.path.join(simulation_path, "band.conf")

    write_force_constants(fc_mat, force_constants_path)
    write_band_conf(band_conf_path, atomic_masses, poscar_path)

    subprocess.run(
        [args.phonopy_command, "-p", "-s", "band.conf"],
        cwd=simulation_path,
        check=True,
    )

    generation_tag = f"{args.generation:06d}"

    band_yaml_src = os.path.join(simulation_path, "band.yaml")
    band_pdf_src = os.path.join(simulation_path, "band.pdf")

    band_yaml_dst = os.path.join(simulation_path, f"band_generation_{generation_tag}.yaml")
    band_pdf_dst = os.path.join(simulation_path, f"band_generation_{generation_tag}.pdf")

    if os.path.isfile(band_yaml_src):
        os.replace(band_yaml_src, band_yaml_dst)
    else:
        raise FileNotFoundError(f"Expected band.yaml was not created in {simulation_path}")

    if os.path.isfile(band_pdf_src):
        os.replace(band_pdf_src, band_pdf_dst)

    phonopy_yaml = os.path.join(simulation_path, "phonopy.yaml")
    if os.path.isfile(phonopy_yaml):
        os.remove(phonopy_yaml)

    q_distance, frequencies = read_band_yaml(band_yaml_dst)

    dispersion_npz_path = os.path.join(
        simulation_path,
        f"dispersion_generation_{generation_tag}.npz",
    )

    np.savez_compressed(
        dispersion_npz_path,
        a_val=a_val,
        atomic_masses=atomic_masses,
        solution=solution,
        generation=args.generation,
        q_distance=q_distance,
        frequencies=frequencies,
    )

    print(f"Wrote {band_yaml_dst}")
    print(f"Wrote {dispersion_npz_path}")


if __name__ == "__main__":
    main()

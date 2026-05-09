#!/usr/bin/env python3
"""
Generate phonon dispersions from MOGA output.

This script can process either:

1. One simulation directory:
   python dispersion_generator_from_simulation.py --simulation simulations/simulation000335

2. All simulations listed in an experiment file:
   python dispersion_generator_from_simulation.py --experiment experiment000007

For each simulation, the script:
- reads a_val and atomic_masses from inputc
- reads one 5-parameter BvK force-constant vector per generation from generation_output_mod.txt
- builds the full BCC force-constant tensor in memory
- passes force constants directly to Phonopy using the Python API
- writes one band_ga.yaml and one dispersion.png per MOGA generation

Default output layout:

simulations/simulationXXXXX/phonon_generations/generation_000001/band_ga.yaml
simulations/simulationXXXXX/phonon_generations/generation_000001/dispersion.png
"""

import argparse
import re
from math import sqrt, isclose
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def read_inputc(input_path):
    """Read a_val and atomic_masses from an inputc file."""
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

    if "a_val" not in params:
        raise KeyError(f"Could not find a_val in {input_path}")

    if "atomic_masses" not in params:
        raise KeyError(f"Could not find atomic_masses in {input_path}")

    a_val = float(params["a_val"])

    atomic_masses = [
        float(x)
        for x in re.split(r"[,\s]+", params["atomic_masses"])
        if x
    ]

    return a_val, np.array(atomic_masses)


def read_generation_output(generation_output_path):
    """
    Read all Best solution vectors from generation_output_mod.txt.

    Expected format:

    Generation 1:
        Best solution: [ alpha0 alpha1 beta1 alpha2 beta2 ]
        Fitness value: [...]
        Index: ...
    """
    generations = []
    best_solutions = []

    current_generation = None

    with open(generation_output_path, "r") as f:
        for line in f:
            line = line.strip()

            gen_match = re.match(r"Generation\s+(\d+):", line)
            if gen_match:
                current_generation = int(gen_match.group(1))
                continue

            if "Best solution" in line:
                match = re.search(r"\[([^\]]+)\]", line)
                if match is None:
                    continue

                values = [
                    float(x)
                    for x in re.split(r"[,\s]+", match.group(1).strip())
                    if x
                ]

                if len(values) != 5:
                    raise ValueError(
                        f"Expected 5 force constants, got {len(values)} "
                        f"in generation {current_generation}: {values}"
                    )

                if current_generation is None:
                    current_generation = len(generations) + 1

                generations.append(current_generation)
                best_solutions.append(values)

    if len(best_solutions) == 0:
        raise ValueError(f"No Best solution entries found in {generation_output_path}")

    return np.array(generations, dtype=int), np.array(best_solutions, dtype=float)


def read_experiment_file(experiment_path):
    """Read simulation names from an experiment file."""
    simulation_names = []

    with open(experiment_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("simulation"):
                simulation_names.append(line)

    if len(simulation_names) == 0:
        raise ValueError(f"No simulation entries found in {experiment_path}")

    return simulation_names


def build_bcc_positions(a_val, system_size):
    """Build ideal positions for a conventional BCC supercell."""
    positions = []
    basis = np.eye(3) * a_val

    for base_atoms in [
        np.array([[0.0, 0.0, 0.0]]) * a_val,
        np.array([[0.5, 0.5, 0.5]]) * a_val,
    ]:
        for i in range(system_size):
            for j in range(system_size):
                for k in range(system_size):
                    base_position = np.array([k, j, i])
                    cart_position = np.inner(basis.T, base_position)

                    for atom in base_atoms:
                        positions.append(cart_position + atom)

    return np.array(positions)


def build_force_constants(a_val, system_size, fc5):
    """
    Build full BCC force-constant tensor from a 5-parameter reduced BvK vector.

    fc5 = [alpha0, alpha1, beta1, alpha2, beta2]

    Remaining 3NN to 5NN force constants are kept at zero.
    """
    alpha0, alpha1, beta1, alpha2, beta2 = fc5

    n = 2 * system_size**3
    alat = a_val * system_size

    ide_lat = build_bcc_positions(a_val, system_size)

    ideal_distances = np.zeros((n, 3, n))
    ideal_dist_sca = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                ideal_distances[i, :, j] = ide_lat[j, :] - ide_lat[i, :]

                # Minimum image convention
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
            if i != j:
                for prox in range(1, 6):
                    if isclose(
                        ideal_dist_sca[i, j],
                        nn_dist[prox - 1],
                        rel_tol=0.001,
                    ):
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

    # Force constants beyond 2NN are fixed to zero.
    alpha3 = 0.0
    beta3 = 0.0
    gamma3 = 0.0
    alpha4 = 0.0
    beta4 = 0.0
    gamma4 = 0.0
    delta4 = 0.0
    alpha5 = 0.0
    beta5 = 0.0

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
    """Optional debugging helper: write Phonopy FORCE_CONSTANTS format."""
    n = fc_mat.shape[0]

    with open(output_path, "w") as file:
        print(f"{n} {n}", file=file)

        for i in range(n):
            for j in range(n):
                print(f"{i + 1} {j + 1}", file=file)
                np.savetxt(file, fc_mat[i, j, :, :])


def make_band_segment(q_start, q_end, n_points):
    q_start = np.array(q_start, dtype=float)
    q_end = np.array(q_end, dtype=float)

    return [
        (1.0 - t) * q_start + t * q_end
        for t in np.linspace(0.0, 1.0, n_points)
    ]


def run_phonopy_in_memory(fc_mat, a_val, mass, system_size, band_points=201):
    """
    Run Phonopy band structure from an in-memory force-constant tensor.
    """
    unitcell = PhonopyAtoms(
        symbols=["Fe", "Fe"],
        cell=[
            [a_val, 0.0, 0.0],
            [0.0, a_val, 0.0],
            [0.0, 0.0, a_val],
        ],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        masses=[mass, mass],
    )

    phonon = Phonopy(
        unitcell,
        supercell_matrix=[
            [system_size, 0, 0],
            [0, system_size, 0],
            [0, 0, system_size],
        ],
        primitive_matrix="auto",
    )

    phonon.force_constants = fc_mat

    bands = [
        make_band_segment([0.0, 0.0, 0.0], [-0.5, 0.5, 0.5], band_points),
        make_band_segment([-0.5, 0.5, 0.5], [0.25, 0.25, 0.25], band_points),
        make_band_segment([0.25, 0.25, 0.25], [0.0, 0.0, 0.0], band_points),
        make_band_segment([0.0, 0.0, 0.0], [0.0, 0.5, 0.0], band_points),
    ]

    phonon.run_band_structure(
        bands,
        path_connections=[True, True, True, False],
        labels=[r"$\Gamma$", "H", "P", r"$\Gamma$", "N"],
    )

    band_dict = phonon.get_band_structure_dict()

    distances = []
    qpoints = []
    frequencies = []

    for segment_distances, segment_qpoints, segment_frequencies in zip(
        band_dict["distances"],
        band_dict["qpoints"],
        band_dict["frequencies"],
    ):
        for d, q, f in zip(segment_distances, segment_qpoints, segment_frequencies):
            distances.append(float(d))
            qpoints.append([float(x) for x in q])
            frequencies.append([float(x) for x in f[:3]])

    return distances, qpoints, frequencies


def write_band_yaml(output_path, distances, qpoints, frequencies, generation, fc5, a_val, mass):
    """
    Write a compact band_ga.yaml file with enough information for downstream ETL.
    """
    data = {
        "generation": int(generation),
        "a_val": float(a_val),
        "mass": float(mass),
        "force_constants_5": {
            "alpha0": float(fc5[0]),
            "alpha1": float(fc5[1]),
            "beta1": float(fc5[2]),
            "alpha2": float(fc5[3]),
            "beta2": float(fc5[4]),
        },
        "phonon": [],
    }

    for d, q, freqs in zip(distances, qpoints, frequencies):
        data["phonon"].append(
            {
                "q-position": q,
                "distance": float(d),
                "band": [
                    {"frequency": float(freqs[0])},
                    {"frequency": float(freqs[1])},
                    {"frequency": float(freqs[2])},
                ],
            }
        )

    with open(output_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def plot_dispersion(
    distances,
    frequencies,
    mass,
    a_val,
    generation,
    band_points=201,
    save_path=None,
    show_plot=False,
):
    freqs = np.array(frequencies)

    plt.figure(figsize=(5.5, 4))

    for branch in range(freqs.shape[1]):
        plt.plot(distances, freqs[:, branch], color="red", linewidth=1.5)

    plt.ylim(-2, 10)
    plt.ylabel("Frequency (THz)", fontsize=20)

    expected_points = 4 * band_points
    if len(distances) < expected_points:
        raise ValueError(
            f"Expected at least {expected_points} distance points, got {len(distances)}"
        )

    x_ticks = [
        distances[0],
        distances[band_points - 1],
        distances[2 * band_points - 1],
        distances[3 * band_points - 1],
        distances[4 * band_points - 1],
    ]

    x_labels = [r"$\Gamma$", "H", "P", r"$\Gamma$", "N"]

    plt.xticks(x_ticks, x_labels, fontsize=20)
    plt.yticks(fontsize=20)

    for x in x_ticks:
        plt.axvline(x=x, color="gray", linestyle=":", linewidth=1)

    plt.axhline(y=0, color="gray", linestyle=":", linewidth=1)

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_visible(True)

    text = (
        f"gen = {generation}\n"
        f"m = {mass:g}\n"
        f"a = {a_val:g}"
    )

    plt.text(
        0.05,
        0.99,
        text,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="top",
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close()


def process_simulation(simulation_path, args):
    """
    Process one simulation directory.
    """
    simulation_path = Path(simulation_path).resolve()

    input_path = simulation_path / args.input_filename
    generation_output_path = simulation_path / args.generation_output_filename
    output_root = simulation_path / args.output_dirname

    if not simulation_path.is_dir():
        print(f"WARNING: Skipping missing simulation directory: {simulation_path}")
        return 0

    if not input_path.is_file():
        print(f"WARNING: Skipping {simulation_path.name}; missing {input_path.name}")
        return 0

    if not generation_output_path.is_file():
        print(
            f"WARNING: Skipping {simulation_path.name}; "
            f"missing {generation_output_path.name}"
        )
        return 0

    a_val, atomic_masses = read_inputc(input_path)

    if len(atomic_masses) == 0:
        print(f"WARNING: Skipping {simulation_path.name}; no atomic masses found.")
        return 0

    mass = float(atomic_masses[0])

    generations, best_solutions = read_generation_output(generation_output_path)

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\nSimulation: {simulation_path.name}")
    print(f"  path = {simulation_path}")
    print(f"  a_val = {a_val}")
    print(f"  mass = {mass:g}")
    print(f"  generations = {len(generations)}")
    print(f"  output = {output_root}")

    n_written = 0

    for generation, fc5 in zip(generations, best_solutions):
        generation_dir = output_root / f"generation_{generation:06d}"
        generation_dir.mkdir(parents=True, exist_ok=True)

        if args.verbose:
            print(f"  Generation {generation}: {fc5}")

        fc_mat = build_force_constants(
            a_val=a_val,
            system_size=args.system_size,
            fc5=fc5,
        )

        if args.write_force_constants:
            force_constants_path = generation_dir / "FORCE_CONSTANTS"
            write_force_constants(fc_mat, force_constants_path)

        distances, qpoints, frequencies = run_phonopy_in_memory(
            fc_mat=fc_mat,
            a_val=a_val,
            mass=mass,
            system_size=args.system_size,
            band_points=args.band_points,
        )

        band_yaml_path = generation_dir / "band_ga.yaml"
        dispersion_path = generation_dir / "dispersion.png"

        write_band_yaml(
            output_path=band_yaml_path,
            distances=distances,
            qpoints=qpoints,
            frequencies=frequencies,
            generation=generation,
            fc5=fc5,
            a_val=a_val,
            mass=mass,
        )

        plot_dispersion(
            distances=distances,
            frequencies=frequencies,
            mass=mass,
            a_val=a_val,
            generation=generation,
            band_points=args.band_points,
            save_path=dispersion_path,
            show_plot=args.show,
        )

        n_written += 1

    print(f"  Wrote {n_written} generation directories.")

    return n_written


def resolve_repo_root(repo_root_arg):
    """
    Resolve repo root.

    Default is the directory containing this script, which works if the script lives
    at the top level of MOGA-Phonons.
    """
    if repo_root_arg is None:
        return Path(__file__).resolve().parent

    return Path(repo_root_arg).resolve()


def resolve_experiment_path(experiment_arg, repo_root):
    """
    Accept either:
    - experiment000007
    - experiments/experiment000007
    - /absolute/path/to/experiment000007
    """
    experiment_path = Path(experiment_arg)

    if experiment_path.is_file():
        return experiment_path.resolve()

    candidate = repo_root / "experiments" / experiment_arg

    if candidate.is_file():
        return candidate.resolve()

    candidate = repo_root / experiment_arg

    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        "Experiment file not found. Tried:\n"
        f"  {experiment_path}\n"
        f"  {repo_root / 'experiments' / experiment_arg}\n"
        f"  {repo_root / experiment_arg}"
    )


def build_simulation_paths_from_args(args):
    """
    Build a list of simulation paths from either --simulation or --experiment.
    """
    repo_root = resolve_repo_root(args.repo_root)

    if args.simulation is not None:
        simulation_path = Path(args.simulation)

        if simulation_path.is_dir():
            return [simulation_path.resolve()]

        candidate = repo_root / args.simulation
        if candidate.is_dir():
            return [candidate.resolve()]

        candidate = repo_root / "simulations" / args.simulation
        if candidate.is_dir():
            return [candidate.resolve()]

        raise NotADirectoryError(
            "Simulation directory not found. Tried:\n"
            f"  {simulation_path}\n"
            f"  {repo_root / args.simulation}\n"
            f"  {repo_root / 'simulations' / args.simulation}"
        )

    experiment_path = resolve_experiment_path(args.experiment, repo_root)
    simulation_names = read_experiment_file(experiment_path)

    print(f"Experiment: {experiment_path}")
    print(f"Found {len(simulation_names)} simulations in experiment file.")

    return [repo_root / "simulations" / name for name in simulation_names]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate phonon dispersions from MOGA generation outputs. "
            "Use either --simulation for one simulation or --experiment for all "
            "simulations listed in an experiment file."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--simulation",
        type=str,
        help=(
            "Path or name of one simulation directory, e.g. "
            "simulations/simulation000335 or simulation000335."
        ),
    )

    mode.add_argument(
        "--experiment",
        type=str,
        help=(
            "Experiment filename or path, e.g. experiment000007 "
            "or experiments/experiment000007."
        ),
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help=(
            "Root path of MOGA-Phonons. Default: directory containing this script."
        ),
    )

    parser.add_argument(
        "--input_filename",
        type=str,
        default="inputc",
        help="Name of input file inside each simulation directory.",
    )

    parser.add_argument(
        "--generation_output_filename",
        type=str,
        default="generation_output_mod.txt",
        help="Name of MOGA generation output file inside each simulation directory.",
    )

    parser.add_argument(
        "--output_dirname",
        type=str,
        default="phonon_generations",
        help="Directory created inside each simulation path for per-generation outputs.",
    )

    parser.add_argument(
        "--system_size",
        type=int,
        default=5,
        help="BCC supercell size. Default: 5, giving 250 atoms.",
    )

    parser.add_argument(
        "--band_points",
        type=int,
        default=201,
        help="Number of q-points per band segment.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show each plot interactively. Usually not recommended for many generations.",
    )

    parser.add_argument(
        "--write_force_constants",
        action="store_true",
        help="Also write FORCE_CONSTANTS to each generation directory.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each generation's force constants while processing.",
    )

    args = parser.parse_args()

    simulation_paths = build_simulation_paths_from_args(args)

    total_written = 0

    for simulation_path in simulation_paths:
        total_written += process_simulation(simulation_path, args)

    print(f"\nDone. Wrote outputs for {total_written} generations.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np


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

    atomic_masses = [
        float(x)
        for x in re.split(r"[,\s]+", params["atomic_masses"])
        if x
    ]

    return a_val, np.array(atomic_masses)


def read_generation_output(generation_output_path):
    best_solutions = []

    with open(generation_output_path, "r") as f:
        for line in f:
            if "Best solution" not in line:
                continue

            match = re.search(r"\[([^\]]+)\]", line)
            if match is None:
                continue

            values = [
                float(x)
                for x in re.split(r"[,\s]+", match.group(1).strip())
                if x
            ]

            best_solutions.append(values)

    return np.array(best_solutions)


def etl_simulation(
    simulation_path,
    input_filename="inputc",
    generation_output_filename="generation_output_mod.txt",
    output_filename="moga_summary.npz",
    verbose=False,
):
    input_path = os.path.join(simulation_path, input_filename)
    generation_output_path = os.path.join(simulation_path, generation_output_filename)
    output_path = os.path.join(simulation_path, output_filename)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Missing input file: {input_path}")

    if not os.path.isfile(generation_output_path):
        raise FileNotFoundError(f"Missing generation output file: {generation_output_path}")

    a_val, atomic_masses = read_inputc(input_path)
    best_solutions = read_generation_output(generation_output_path)
    generations = np.arange(best_solutions.shape[0])

    np.savez_compressed(
        output_path,
        a_val=a_val,
        atomic_masses=atomic_masses,
        generations=generations,
        best_solutions=best_solutions,
    )

    if verbose:
        print(f"Wrote {output_path}")
        print(f"  a_val = {a_val}")
        print(f"  atomic_masses = {atomic_masses}")
        print(f"  best_solutions shape = {best_solutions.shape}")

    return output_path


def read_experiment_file(experiment_path):
    simulation_names = []

    with open(experiment_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("simulation"):
                simulation_names.append(line)

    return simulation_names


def etl_experiment(
    experiment_name,
    repo_root=None,
    input_filename="inputc",
    generation_output_filename="generation_output_mod.txt",
    output_filename="moga_summary.npz",
    verbose=False,
):
    user = os.environ["USER"]

    if repo_root is None:
        repo_root = f"/home/{user}/MOGA-Phonons"

    experiment_path = os.path.join(repo_root, "experiments", experiment_name)
    simulations_root = os.path.join(repo_root, "simulations")

    if not os.path.isfile(experiment_path):
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    simulation_names = read_experiment_file(experiment_path)

    if verbose:
        print(f"Experiment: {experiment_name}")
        print(f"Found {len(simulation_names)} simulations.")

    output_paths = []

    for sim_name in simulation_names:
        simulation_path = os.path.join(simulations_root, sim_name)

        try:
            output_path = etl_simulation(
                simulation_path=simulation_path,
                input_filename=input_filename,
                generation_output_filename=generation_output_filename,
                output_filename=output_filename,
                verbose=verbose,
            )
            output_paths.append(output_path)

        except FileNotFoundError as err:
            print(f"WARNING: Skipping {sim_name}")
            print(f"  {err}")

    print(f"\nDone. Wrote {len(output_paths)} npz files.\n")

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create compressed npz summaries for all simulations in a MOGA experiment."
    )

    parser.add_argument(
        "experiment_name",
        type=str,
        help="Experiment filename, e.g. experiment000000",
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Root path of the MOGA-Phonons repo. Default: /home/$USER/MOGA-Phonons",
    )

    parser.add_argument(
        "--input_filename",
        type=str,
        default="inputc",
        help="Input filename inside each simulation directory.",
    )

    parser.add_argument(
        "--generation_output_filename",
        type=str,
        default="generation_output_mod.txt",
        help="Generation output filename inside each simulation directory.",
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        default="moga_summary.npz",
        help="Output npz filename written inside each simulation directory.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print details while processing.",
    )

    args = parser.parse_args()

    etl_experiment(
        experiment_name=args.experiment_name,
        repo_root=args.repo_root,
        input_filename=args.input_filename,
        generation_output_filename=args.generation_output_filename,
        output_filename=args.output_filename,
        verbose=args.verbose,
    )

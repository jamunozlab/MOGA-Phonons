#!/usr/bin/env python3

import os
import shutil
import argparse


def move_experiment(
    experiment_name,
    repo_root=None,
    scratch_root=None,
    verbose=True,
):
    """
    Move one experiment file and all associated simulations to scratch.

    Expected experiment file format:

        Description line(s)

        simulation000000
        simulation000001
        ...

    Example:
        python move_experiment_to_scratch.py experiment000000
    """

    user = os.environ["USER"]

    if repo_root is None:
        repo_root = f"/home/{user}/MOGA-Phonons"

    if scratch_root is None:
        scratch_root = f"/scratch/{user}/MOGA-Phonons"

    experiments_src = os.path.join(repo_root, "experiments")
    simulations_src = os.path.join(repo_root, "simulations")

    experiments_dst = os.path.join(scratch_root, "experiments")
    simulations_dst = os.path.join(scratch_root, "simulations")

    os.makedirs(experiments_dst, exist_ok=True)
    os.makedirs(simulations_dst, exist_ok=True)

    experiment_path = os.path.join(experiments_src, experiment_name)

    if not os.path.isfile(experiment_path):
        raise FileNotFoundError(f"Experiment file not found:\n{experiment_path}")

    if verbose:
        print(f"\nReading experiment file:\n{experiment_path}\n")

    simulation_names = []

    with open(experiment_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("simulation"):
                simulation_names.append(line)

    if verbose:
        print(f"Found {len(simulation_names)} simulations.\n")

    # ------------------------------------------------------------------
    # Move experiment file
    # ------------------------------------------------------------------

    experiment_dst_path = os.path.join(experiments_dst, experiment_name)

    if verbose:
        print(f"Moving experiment file:")
        print(f"  {experiment_path}")
        print(f"  ->")
        print(f"  {experiment_dst_path}\n")

    shutil.move(experiment_path, experiment_dst_path)

    # ------------------------------------------------------------------
    # Move simulations
    # ------------------------------------------------------------------

    for sim_name in simulation_names:

        sim_src = os.path.join(simulations_src, sim_name)
        sim_dst = os.path.join(simulations_dst, sim_name)

        if not os.path.exists(sim_src):
            print(f"WARNING: Missing simulation directory: {sim_src}")
            continue

        if verbose:
            print(f"Moving:")
            print(f"  {sim_src}")
            print(f"  ->")
            print(f"  {sim_dst}\n")

        shutil.move(sim_src, sim_dst)

    print("\nDone.\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Move one MOGA experiment and its simulations to scratch."
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
        help="Root MOGA repo path",
    )

    parser.add_argument(
        "--scratch_root",
        type=str,
        default=None,
        help="Scratch destination root",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    move_experiment(
        experiment_name=args.experiment_name,
        repo_root=args.repo_root,
        scratch_root=args.scratch_root,
        verbose=not args.quiet,
    )

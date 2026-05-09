#!/usr/bin/env python3
"""
Build a Pandas dataframe from all per-generation phonon band YAML files
associated with a MOGA experiment.

Example
-------
From the MOGA-Phonons repository root:

    python build_experiment_dataframe.py --experiment experiment000007

This writes:

    dataframes/dataframe000007.pkl

The dataframe contains one row per simulation/generation pair, including:
- simulation name
- generation number
- mass
- lattice parameter
- five reduced BvK force constants
- q-path distances
- q-points
- phonon frequencies
- simple stability metrics
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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


def experiment_number_from_name(experiment_name):
    """Extract numeric suffix from experiment name: experiment000007 -> 000007."""
    match = re.search(r"(\d+)$", experiment_name)

    if match is None:
        raise ValueError(
            f"Could not extract experiment number from {experiment_name}. "
            "Use --output_filename explicitly."
        )

    return match.group(1)


def resolve_repo_root(repo_root_arg):
    """Default repo root is the directory containing this script."""
    if repo_root_arg is None:
        return Path(__file__).resolve().parent

    return Path(repo_root_arg).resolve()


def resolve_experiment_path(experiment_arg, repo_root):
    """
    Accept:
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


def load_band_yaml(yaml_path):
    """Load one band_ga.yaml file and return parsed quantities."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    generation = data["generation"]
    a_val = data["a_val"]
    mass = data["mass"]
    fc5 = data["force_constants_5"]
    phonons = data["phonon"]

    distances = []
    qpoints = []
    frequencies = []

    for phonon in phonons:
        distances.append(phonon["distance"])
        qpoints.append(phonon["q-position"])

        band_freqs = [
            band["frequency"]
            for band in phonon["band"]
        ]

        frequencies.append(band_freqs)

    distances = np.array(distances, dtype=float)
    qpoints = np.array(qpoints, dtype=float)
    frequencies = np.array(frequencies, dtype=float)

    return {
        "generation": generation,
        "a_val": a_val,
        "mass": mass,
        "alpha0": fc5["alpha0"],
        "alpha1": fc5["alpha1"],
        "beta1": fc5["beta1"],
        "alpha2": fc5["alpha2"],
        "beta2": fc5["beta2"],
        "distances": distances,
        "qpoints": qpoints,
        "frequencies": frequencies,
        "min_frequency": float(np.min(frequencies)),
        "max_frequency": float(np.max(frequencies)),
        "num_imaginary": int(np.sum(frequencies < 0.0)),
    }


def build_experiment_dataframe(
    experiment_path,
    repo_root,
    simulations_root,
    phonon_dirname,
    band_yaml_filename,
    verbose=False,
):
    """Build dataframe from all band YAML files listed by an experiment file."""
    simulation_names = read_experiment_file(experiment_path)

    rows = []
    missing_phonon_dirs = []
    missing_yaml_files = []

    for simulation_name in simulation_names:
        simulation_path = simulations_root / simulation_name
        phonon_root = simulation_path / phonon_dirname

        if not phonon_root.is_dir():
            missing_phonon_dirs.append(str(phonon_root))
            if verbose:
                print(f"WARNING: Missing {phonon_root}")
            continue

        generation_dirs = sorted(phonon_root.glob("generation_*"))

        if len(generation_dirs) == 0:
            if verbose:
                print(f"WARNING: No generation directories in {phonon_root}")
            continue

        for generation_dir in generation_dirs:
            yaml_path = generation_dir / band_yaml_filename

            if not yaml_path.is_file():
                missing_yaml_files.append(str(yaml_path))
                if verbose:
                    print(f"WARNING: Missing {yaml_path}")
                continue

            row = load_band_yaml(yaml_path)

            row["simulation"] = simulation_name
            row["simulation_path"] = str(simulation_path.relative_to(repo_root))
            row["generation_dir"] = str(generation_dir.relative_to(repo_root))
            row["band_yaml_path"] = str(yaml_path.relative_to(repo_root))

            rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df[
            [
                "simulation",
                "generation",
                "mass",
                "a_val",
                "alpha0",
                "alpha1",
                "beta1",
                "alpha2",
                "beta2",
                "min_frequency",
                "max_frequency",
                "num_imaginary",
                "distances",
                "qpoints",
                "frequencies",
                "simulation_path",
                "generation_dir",
                "band_yaml_path",
            ]
        ]

        df = df.sort_values(["simulation", "generation"]).reset_index(drop=True)

    summary = {
        "n_simulations_in_experiment": len(simulation_names),
        "n_rows": len(df),
        "n_missing_phonon_dirs": len(missing_phonon_dirs),
        "n_missing_yaml_files": len(missing_yaml_files),
        "missing_phonon_dirs": missing_phonon_dirs,
        "missing_yaml_files": missing_yaml_files,
    }

    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load all band_ga.yaml files associated with a MOGA experiment "
            "into a Pandas dataframe and save it as a pickle file."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment000007",
        help=(
            "Experiment filename or path. Examples: experiment000007, "
            "experiments/experiment000007, or an absolute path."
        ),
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Root path of MOGA-Phonons. Default: directory containing this script.",
    )

    parser.add_argument(
        "--simulations_dirname",
        type=str,
        default="simulations",
        help="Name of simulations directory inside repo root.",
    )

    parser.add_argument(
        "--phonon_dirname",
        type=str,
        default="phonon_generations",
        help="Name of per-simulation phonon output directory.",
    )

    parser.add_argument(
        "--band_yaml_filename",
        type=str,
        default="band_ga.yaml",
        help="Name of band YAML file inside each generation directory.",
    )

    parser.add_argument(
        "--dataframes_dirname",
        type=str,
        default="dataframes",
        help="Directory inside repo root where pickle file is saved.",
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help=(
            "Output pickle filename. Default is dataframeXXXXXX.pkl, "
            "using the numeric suffix of the experiment name."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print missing files/directories while loading.",
    )

    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    experiment_path = resolve_experiment_path(args.experiment, repo_root)

    simulations_root = repo_root / args.simulations_dirname
    dataframes_root = repo_root / args.dataframes_dirname
    dataframes_root.mkdir(parents=True, exist_ok=True)

    if args.output_filename is None:
        experiment_number = experiment_number_from_name(experiment_path.name)
        output_filename = f"dataframe{experiment_number}.pkl"
    else:
        output_filename = args.output_filename

    output_path = dataframes_root / output_filename

    print(f"Repo root: {repo_root}")
    print(f"Experiment: {experiment_path}")
    print(f"Simulations root: {simulations_root}")
    print(f"Output pickle: {output_path}")

    df, summary = build_experiment_dataframe(
        experiment_path=experiment_path,
        repo_root=repo_root,
        simulations_root=simulations_root,
        phonon_dirname=args.phonon_dirname,
        band_yaml_filename=args.band_yaml_filename,
        verbose=args.verbose,
    )

    df.to_pickle(output_path)

    print("\nSummary")
    print("-------")
    print(f"Simulations in experiment: {summary['n_simulations_in_experiment']}")
    print(f"Dataframe rows: {summary['n_rows']}")
    print(f"Missing phonon directories: {summary['n_missing_phonon_dirs']}")
    print(f"Missing YAML files: {summary['n_missing_yaml_files']}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

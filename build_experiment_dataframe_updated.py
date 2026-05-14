#!/usr/bin/env python3
"""
Build a Pandas dataframe from phonon dispersion YAML files associated with a
MOGA experiment.

This version supports both output layouts:

1. Best-only layout
   simulationXXXXXX/phonon_generations/generation_YYYYYY/band_ga.yaml

2. Multi-solution layout, e.g. top-k or all
   simulationXXXXXX/phonon_generations/generation_YYYYYY/solution_ZZZZZZ_rank_RRR/band_ga.yaml

Example
-------
From the MOGA-Phonons repository root:

    python build_experiment_dataframe.py --experiment experiment000007

This writes:

    dataframes/dataframe000007.pkl
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_experiment_file(experiment_path):
    """Read simulation names from a plain-text experiment file."""
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


def safe_relative_path(path, root):
    """Return path relative to root when possible; otherwise return absolute path."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_json_if_exists(path):
    """Load a JSON file if present; otherwise return None."""
    if path is None or not Path(path).is_file():
        return None
    with open(path, "r") as f:
        return json.load(f)


def parse_solution_dir_metadata(yaml_path):
    """
    Infer solution index and rank from path components such as:
        solution_000017_rank_000
    """
    solution_index_from_path = None
    rank_from_path = None

    for part in yaml_path.parts:
        match = re.match(r"solution_(\d+)(?:_rank_(\d+))?$", part)
        if match:
            solution_index_from_path = int(match.group(1))
            if match.group(2) is not None:
                rank_from_path = int(match.group(2))
            break

    return solution_index_from_path, rank_from_path


def load_band_yaml(yaml_path):
    """Load one band_ga.yaml file and return parsed quantities."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    generation = data["generation"]
    a_val = data["a_val"]
    mass = data["mass"]
    fc5 = data["force_constants_5"]
    phonons = data["phonon"]

    solution_index_from_path, rank_from_path = parse_solution_dir_metadata(yaml_path)
    solution_index = data.get("solution_index", solution_index_from_path)
    rank = data.get("rank", rank_from_path)

    fitness = data.get("fitness", None)
    fitness_norm = data.get("fitness_norm", None)

    if fitness is not None:
        fitness = [float(x) for x in fitness]
        fitness1 = fitness[0] if len(fitness) > 0 else np.nan
        fitness2 = fitness[1] if len(fitness) > 1 else np.nan
        fitness3 = fitness[2] if len(fitness) > 2 else np.nan
    else:
        fitness1 = np.nan
        fitness2 = np.nan
        fitness3 = np.nan

    if fitness_norm is None and fitness is not None:
        fitness_norm = float(np.linalg.norm(fitness))

    distances = []
    qpoints = []
    frequencies = []

    for phonon in phonons:
        distances.append(phonon["distance"])
        qpoints.append(phonon["q-position"])
        band_freqs = [band["frequency"] for band in phonon["band"]]
        frequencies.append(band_freqs)

    distances = np.array(distances, dtype=float)
    qpoints = np.array(qpoints, dtype=float)
    frequencies = np.array(frequencies, dtype=float)

    min_frequency = float(np.min(frequencies))
    max_frequency = float(np.max(frequencies))
    num_imaginary = int(np.sum(frequencies < 0.0))

    acoustic_gamma = frequencies[0, :3].tolist() if frequencies.shape[1] >= 3 else []
    optical_gamma = frequencies[0, 3:].tolist() if frequencies.shape[1] > 3 else []

    return {
        "generation": int(generation),
        "solution_index": None if solution_index is None else int(solution_index),
        "rank": None if rank is None else int(rank),
        "a_val": float(a_val),
        "mass": float(mass),
        "alpha0": float(fc5["alpha0"]),
        "alpha1": float(fc5["alpha1"]),
        "beta1": float(fc5["beta1"]),
        "alpha2": float(fc5["alpha2"]),
        "beta2": float(fc5["beta2"]),
        "fitness": fitness,
        "fitness1": fitness1,
        "fitness2": fitness2,
        "fitness3": fitness3,
        "fitness_norm": np.nan if fitness_norm is None else float(fitness_norm),
        "distances": distances,
        "qpoints": qpoints,
        "frequencies": frequencies,
        "min_frequency": min_frequency,
        "max_frequency": max_frequency,
        "num_imaginary": num_imaginary,
        "is_stable": bool(num_imaginary == 0),
        "gamma_acoustic_frequencies": acoustic_gamma,
        "gamma_optical_frequencies": optical_gamma,
    }


def discover_band_yaml_files(phonon_root, band_yaml_filename, include_nested=True):
    """
    Discover band YAML files in a phonon_generations directory.

    Supports:
    - generation_XXXXXX/band_ga.yaml
    - generation_XXXXXX/solution_.../band_ga.yaml
    """
    yaml_paths = []
    generation_dirs = sorted(phonon_root.glob("generation_*"))

    for generation_dir in generation_dirs:
        direct_yaml = generation_dir / band_yaml_filename
        if direct_yaml.is_file():
            yaml_paths.append(direct_yaml)

        if include_nested:
            nested_paths = sorted(generation_dir.glob(f"solution_*/{band_yaml_filename}"))
            yaml_paths.extend(nested_paths)

    return sorted(set(yaml_paths))


def build_experiment_dataframe(
    experiment_path,
    repo_root,
    simulations_root,
    phonon_dirname,
    band_yaml_filename,
    include_nested=True,
    experiment_manifest_path=None,
    verbose=False,
):
    """Build dataframe from all band YAML files listed by an experiment file."""
    simulation_names = read_experiment_file(experiment_path)
    experiment_manifest = load_json_if_exists(experiment_manifest_path)

    rows = []
    missing_simulation_dirs = []
    missing_phonon_dirs = []
    simulations_with_no_yaml = []
    load_errors = []

    for simulation_name in simulation_names:
        simulation_path = simulations_root / simulation_name

        if not simulation_path.is_dir():
            missing_simulation_dirs.append(str(simulation_path))
            if verbose:
                print(f"WARNING: Missing simulation directory {simulation_path}")
            continue

        phonon_root = simulation_path / phonon_dirname

        if not phonon_root.is_dir():
            missing_phonon_dirs.append(str(phonon_root))
            if verbose:
                print(f"WARNING: Missing {phonon_root}")
            continue

        yaml_paths = discover_band_yaml_files(
            phonon_root=phonon_root,
            band_yaml_filename=band_yaml_filename,
            include_nested=include_nested,
        )

        if len(yaml_paths) == 0:
            simulations_with_no_yaml.append(str(phonon_root))
            if verbose:
                print(f"WARNING: No {band_yaml_filename} files in {phonon_root}")
            continue

        for yaml_path in yaml_paths:
            try:
                row = load_band_yaml(yaml_path)
            except Exception as exc:
                load_errors.append({"path": str(yaml_path), "error": repr(exc)})
                if verbose:
                    print(f"WARNING: Failed to load {yaml_path}: {exc}")
                continue

            row["simulation"] = simulation_name
            row["simulation_path"] = safe_relative_path(simulation_path, repo_root)
            row["phonon_root"] = safe_relative_path(phonon_root, repo_root)
            row["generation_dir"] = safe_relative_path(yaml_path.parent, repo_root)
            row["band_yaml_path"] = safe_relative_path(yaml_path, repo_root)

            if experiment_manifest is not None:
                row["experiment_manifest_path"] = safe_relative_path(
                    Path(experiment_manifest_path), repo_root
                )
                row["experiment_manifest"] = experiment_manifest

            rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        preferred_columns = [
            "simulation",
            "generation",
            "solution_index",
            "rank",
            "mass",
            "a_val",
            "alpha0",
            "alpha1",
            "beta1",
            "alpha2",
            "beta2",
            "fitness",
            "fitness1",
            "fitness2",
            "fitness3",
            "fitness_norm",
            "min_frequency",
            "max_frequency",
            "num_imaginary",
            "is_stable",
            "gamma_acoustic_frequencies",
            "gamma_optical_frequencies",
            "distances",
            "qpoints",
            "frequencies",
            "simulation_path",
            "phonon_root",
            "generation_dir",
            "band_yaml_path",
            "experiment_manifest_path",
            "experiment_manifest",
        ]

        existing_preferred_columns = [c for c in preferred_columns if c in df.columns]
        remaining_columns = [c for c in df.columns if c not in existing_preferred_columns]
        df = df[existing_preferred_columns + remaining_columns]

        sort_columns = ["simulation", "generation"]
        if "rank" in df.columns:
            sort_columns.append("rank")
        if "solution_index" in df.columns:
            sort_columns.append("solution_index")

        df = df.sort_values(sort_columns).reset_index(drop=True)

    summary = {
        "n_simulations_in_experiment": len(simulation_names),
        "n_rows": len(df),
        "n_missing_simulation_dirs": len(missing_simulation_dirs),
        "n_missing_phonon_dirs": len(missing_phonon_dirs),
        "n_simulations_with_no_yaml": len(simulations_with_no_yaml),
        "n_load_errors": len(load_errors),
        "missing_simulation_dirs": missing_simulation_dirs,
        "missing_phonon_dirs": missing_phonon_dirs,
        "simulations_with_no_yaml": simulations_with_no_yaml,
        "load_errors": load_errors,
    }

    return df, summary


def write_summary_json(summary, output_path):
    """Write build summary next to dataframe for provenance/debugging."""
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load all band_ga.yaml files associated with a MOGA experiment "
            "into a Pandas dataframe and save it as a pickle file."
        )
    )

    parser.add_argument("--experiment", type=str, default="experiment000007")
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--simulations_dirname", type=str, default="simulations")
    parser.add_argument("--phonon_dirname", type=str, default="phonon_generations")
    parser.add_argument("--band_yaml_filename", type=str, default="band_ga.yaml")
    parser.add_argument("--dataframes_dirname", type=str, default="dataframes")
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument(
        "--experiment_manifest",
        type=str,
        default=None,
        help="Optional experiment manifest JSON file to include in each row.",
    )
    parser.add_argument(
        "--no_nested",
        action="store_true",
        help="Disable scanning nested solution directories.",
    )
    parser.add_argument(
        "--write_summary",
        action="store_true",
        help="Write a dataframeXXXXXX.summary.json file next to the pickle.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

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

    experiment_manifest_path = None
    if args.experiment_manifest is not None:
        experiment_manifest_path = Path(args.experiment_manifest)
        if not experiment_manifest_path.is_file():
            candidate = repo_root / args.experiment_manifest
            if candidate.is_file():
                experiment_manifest_path = candidate
            else:
                raise FileNotFoundError(
                    f"Experiment manifest not found: {args.experiment_manifest}"
                )

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
        include_nested=not args.no_nested,
        experiment_manifest_path=experiment_manifest_path,
        verbose=args.verbose,
    )

    df.to_pickle(output_path)

    summary_path = None
    if args.write_summary:
        summary_path = write_summary_json(summary, output_path)

    print("\nSummary")
    print("-------")
    print(f"Simulations in experiment: {summary['n_simulations_in_experiment']}")
    print(f"Dataframe rows: {summary['n_rows']}")
    print(f"Missing simulation directories: {summary['n_missing_simulation_dirs']}")
    print(f"Missing phonon directories: {summary['n_missing_phonon_dirs']}")
    print(f"Simulations with no YAML files: {summary['n_simulations_with_no_yaml']}")
    print(f"YAML load errors: {summary['n_load_errors']}")
    print(f"\nSaved: {output_path}")

    if summary_path is not None:
        print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()

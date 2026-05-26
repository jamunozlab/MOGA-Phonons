#!/usr/bin/env python3
"""
Multiprocessing dataframe builder for MOGA-Phonons band_ga.yaml outputs.

Example
-------
From the MOGA-Phonons repository root:

    python build_experiment_dataframe_mp.py \
        --experiment experiment000011 \
        --nproc 8 \
        --progress \
        --write_summary

This writes by default:

    dataframes/dataframe000011.pkl

Supports both layouts:
1. generation_000001/band_ga.yaml
2. generation_000001/solution_000032_rank_000/band_ga.yaml
"""

import argparse
import json
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml


def read_experiment_file(experiment_path):
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
    match = re.search(r"(\d+)$", experiment_name)

    if match is None:
        raise ValueError(
            f"Could not extract experiment number from {experiment_name}. "
            "Use --output_filename explicitly."
        )

    return match.group(1)


def resolve_repo_root(repo_root_arg):
    if repo_root_arg is None:
        return Path(__file__).resolve().parent
    return Path(repo_root_arg).resolve()


def resolve_experiment_path(experiment_arg, repo_root):
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


def parse_generation_dir(generation_dir_name):
    match = re.search(r"generation_(\d+)", generation_dir_name)
    if match:
        return int(match.group(1))
    return None


def parse_solution_dir(solution_dir_name):
    match = re.search(r"solution_(\d+)_rank_(\d+)", solution_dir_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def collect_yaml_tasks(
    experiment_path,
    repo_root,
    simulations_root,
    phonon_dirname,
    band_yaml_filename,
):
    simulation_names = read_experiment_file(experiment_path)

    tasks = []
    missing_simulation_dirs = []
    missing_phonon_dirs = []
    simulations_with_no_yaml = []

    for simulation_name in simulation_names:
        simulation_path = simulations_root / simulation_name
        phonon_root = simulation_path / phonon_dirname

        if not simulation_path.is_dir():
            missing_simulation_dirs.append(str(simulation_path))
            continue

        if not phonon_root.is_dir():
            missing_phonon_dirs.append(str(phonon_root))
            continue

        yaml_paths = sorted(phonon_root.glob(f"generation_*/{band_yaml_filename}"))
        yaml_paths += sorted(phonon_root.glob(f"generation_*/solution_*/{band_yaml_filename}"))

        if len(yaml_paths) == 0:
            simulations_with_no_yaml.append(str(simulation_path))
            continue

        for yaml_path in yaml_paths:
            generation_dir = None
            solution_dir = None

            if yaml_path.parent.name.startswith("solution_"):
                solution_dir = yaml_path.parent
                generation_dir = yaml_path.parent.parent
            elif yaml_path.parent.name.startswith("generation_"):
                generation_dir = yaml_path.parent

            tasks.append(
                {
                    "yaml_path": str(yaml_path),
                    "simulation_name": simulation_name,
                    "simulation_path": str(simulation_path),
                    "generation_dir": str(generation_dir) if generation_dir is not None else None,
                    "solution_dir": str(solution_dir) if solution_dir is not None else None,
                    "repo_root": str(repo_root),
                }
            )

    summary = {
        "n_simulations_in_experiment": len(simulation_names),
        "n_yaml_tasks": len(tasks),
        "n_missing_simulation_dirs": len(missing_simulation_dirs),
        "n_missing_phonon_dirs": len(missing_phonon_dirs),
        "n_simulations_with_no_yaml": len(simulations_with_no_yaml),
        "missing_simulation_dirs": missing_simulation_dirs,
        "missing_phonon_dirs": missing_phonon_dirs,
        "simulations_with_no_yaml": simulations_with_no_yaml,
    }

    return tasks, summary


def safe_relative(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_band_yaml_task(task):
    yaml_path = Path(task["yaml_path"])
    repo_root = Path(task["repo_root"])
    simulation_path = Path(task["simulation_path"])

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    generation = data.get("generation")
    if generation is None and task["generation_dir"] is not None:
        generation = parse_generation_dir(Path(task["generation_dir"]).name)

    solution_index = data.get("solution_index", None)
    rank = data.get("rank", None)

    if task["solution_dir"] is not None:
        parsed_solution_index, parsed_rank = parse_solution_dir(Path(task["solution_dir"]).name)
        if solution_index is None:
            solution_index = parsed_solution_index
        if rank is None:
            rank = parsed_rank

    a_val = np.float32(data["a_val"])
    mass = np.float32(data["mass"])

    fc5 = data["force_constants_5"]
    selection_metadata = data.get("selection_metadata", {}) or {}

    fitness = data.get("fitness", None)
    if fitness is None:
        fitness = selection_metadata.get("fitness", None)

    fitness_norm = data.get("fitness_norm", None)
    if fitness_norm is None:
        fitness_norm = selection_metadata.get("fitness_norm", None)

    sort_by = selection_metadata.get("sort_by", data.get("sort_by", None))
    sort_score = selection_metadata.get("sort_score", data.get("sort_score", None))

    source = data.get("source", None)
    timestamp = data.get("timestamp", None)
    system_size = data.get("system_size", None)

    distances = []
    qpoints = []
    frequencies = []

    for phonon in data["phonon"]:
        distances.append(phonon["distance"])
        qpoints.append(phonon["q-position"])
        frequencies.append([band["frequency"] for band in phonon["band"]])

    distances = np.array(distances, dtype=np.float32)
    qpoints = np.array(qpoints, dtype=np.float32)
    frequencies = np.array(frequencies, dtype=np.float32)

    if fitness is None:
        fitness_arr = np.array([], dtype=np.float32)
        fitness1 = np.float32(np.nan)
        fitness2 = np.float32(np.nan)
        fitness3 = np.float32(np.nan)
    else:
        fitness_arr = np.array(fitness, dtype=np.float32)
        fitness1 = np.float32(fitness_arr[0]) if fitness_arr.size > 0 else np.float32(np.nan)
        fitness2 = np.float32(fitness_arr[1]) if fitness_arr.size > 1 else np.float32(np.nan)
        fitness3 = np.float32(fitness_arr[2]) if fitness_arr.size > 2 else np.float32(np.nan)

    if fitness_norm is None:
        fitness_norm_value = np.float32(np.linalg.norm(fitness_arr)) if fitness_arr.size else np.float32(np.nan)
    else:
        fitness_norm_value = np.float32(fitness_norm)

    min_frequency = np.float32(np.min(frequencies))
    max_frequency = np.float32(np.max(frequencies))
    num_imaginary = np.int32(np.sum(frequencies < 0.0))
    gamma_frequencies = frequencies[0].astype(np.float32)

    return {
        "simulation": task["simulation_name"],
        "generation": np.int32(generation) if generation is not None else np.int32(-1),
        "solution_index": np.int32(solution_index) if solution_index is not None else np.int32(-1),
        "rank": np.int32(rank) if rank is not None else np.int32(-1),
        "source": source,
        "timestamp": timestamp,
        "mass": mass,
        "a_val": a_val,
        "system_size": np.int32(system_size) if system_size is not None else np.int32(-1),
        "alpha0": np.float32(fc5["alpha0"]),
        "alpha1": np.float32(fc5["alpha1"]),
        "beta1": np.float32(fc5["beta1"]),
        "alpha2": np.float32(fc5["alpha2"]),
        "beta2": np.float32(fc5["beta2"]),
        "fitness": fitness_arr,
        "fitness1": fitness1,
        "fitness2": fitness2,
        "fitness3": fitness3,
        "fitness_norm": fitness_norm_value,
        "sort_by": sort_by,
        "sort_score": np.float32(sort_score) if sort_score is not None else np.float32(np.nan),
        "min_frequency": min_frequency,
        "max_frequency": max_frequency,
        "num_imaginary": num_imaginary,
        "stable": bool(min_frequency >= 0.0),
        "gamma_frequencies": gamma_frequencies,
        "distances": distances,
        "qpoints": qpoints,
        "frequencies": frequencies,
        "frequency_vector": frequencies.reshape(-1).astype(np.float32),
        "simulation_path": safe_relative(simulation_path, repo_root),
        "generation_dir": safe_relative(Path(task["generation_dir"]), repo_root) if task["generation_dir"] else None,
        "solution_dir": safe_relative(Path(task["solution_dir"]), repo_root) if task["solution_dir"] else None,
        "band_yaml_path": safe_relative(yaml_path, repo_root),
    }


def build_dataframe_parallel(tasks, nproc, progress=False, progress_every=100):
    rows = []
    errors = []

    if nproc == 1:
        for i, task in enumerate(tasks, start=1):
            try:
                rows.append(load_band_yaml_task(task))
            except Exception as exc:
                errors.append((task["yaml_path"], repr(exc)))

            if progress and (i % progress_every == 0 or i == len(tasks)):
                print(f"Loaded {i}/{len(tasks)} YAML files")
    else:
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            future_to_path = {
                executor.submit(load_band_yaml_task, task): task["yaml_path"]
                for task in tasks
            }

            for i, future in enumerate(as_completed(future_to_path), start=1):
                path = future_to_path[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    errors.append((path, repr(exc)))

                if progress and (i % progress_every == 0 or i == len(future_to_path)):
                    print(f"Loaded {i}/{len(future_to_path)} YAML files")

    if errors:
        print("\nWARNING: Some YAML files failed to load.")
        for path, err in errors[:20]:
            print(f"  {path}: {err}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more errors not shown.")

    df = pd.DataFrame(rows)

    if len(df) > 0:
        preferred_columns = [
            "simulation", "generation", "solution_index", "rank",
            "source", "timestamp", "mass", "a_val", "system_size",
            "alpha0", "alpha1", "beta1", "alpha2", "beta2",
            "fitness", "fitness1", "fitness2", "fitness3", "fitness_norm",
            "sort_by", "sort_score",
            "min_frequency", "max_frequency", "num_imaginary", "stable",
            "gamma_frequencies", "distances", "qpoints", "frequencies",
            "frequency_vector",
            "simulation_path", "generation_dir", "solution_dir", "band_yaml_path",
        ]
        existing_columns = [c for c in preferred_columns if c in df.columns]
        remaining_columns = [c for c in df.columns if c not in existing_columns]
        df = df[existing_columns + remaining_columns]

        sort_cols = [c for c in ["simulation", "generation", "rank", "solution_index"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df, errors


def write_summary_json(summary_path, summary):
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Multiprocessing loader for MOGA band_ga.yaml files."
    )

    parser.add_argument("--experiment", type=str, default="experiment000007")
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--simulations_dirname", type=str, default="simulations")
    parser.add_argument("--phonon_dirname", type=str, default="phonon_generations")
    parser.add_argument("--band_yaml_filename", type=str, default="band_ga.yaml")
    parser.add_argument("--dataframes_dirname", type=str, default="dataframes")
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--write_summary", action="store_true")

    args = parser.parse_args()

    if args.nproc < 1:
        raise ValueError("--nproc must be >= 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")

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
    summary_path = output_path.with_suffix(".summary.json")

    print(f"Repo root: {repo_root}")
    print(f"Experiment: {experiment_path}")
    print(f"Simulations root: {simulations_root}")
    print(f"Output pickle: {output_path}")
    print(f"nproc: {args.nproc}")

    tasks, collection_summary = collect_yaml_tasks(
        experiment_path=experiment_path,
        repo_root=repo_root,
        simulations_root=simulations_root,
        phonon_dirname=args.phonon_dirname,
        band_yaml_filename=args.band_yaml_filename,
    )

    print(f"YAML files queued: {len(tasks)}")

    df, errors = build_dataframe_parallel(
        tasks=tasks,
        nproc=args.nproc,
        progress=args.progress,
        progress_every=args.progress_every,
    )

    df.to_pickle(output_path)

    summary = {
        **collection_summary,
        "n_rows": int(len(df)),
        "n_load_errors": int(len(errors)),
        "load_errors": errors[:100],
        "output_path": str(output_path),
    }

    print("\nSummary")
    print("-------")
    print(f"Simulations in experiment: {summary['n_simulations_in_experiment']}")
    print(f"YAML files queued: {summary['n_yaml_tasks']}")
    print(f"Dataframe rows: {summary['n_rows']}")
    print(f"Missing simulation directories: {summary['n_missing_simulation_dirs']}")
    print(f"Missing phonon directories: {summary['n_missing_phonon_dirs']}")
    print(f"Simulations with no YAML files: {summary['n_simulations_with_no_yaml']}")
    print(f"YAML load errors: {summary['n_load_errors']}")
    print(f"\nSaved: {output_path}")

    if args.write_summary:
        write_summary_json(summary_path, summary)
        print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_simulation_ids(experiment_path):
    simulation_ids = []

    with open(experiment_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("simulation"):
                simulation_ids.append(line)

    return simulation_ids


def run_one_simulation(sim_dir, conda_env, workers_per_simulation):
    env = os.environ.copy()

    env["SLURM_CPUS_PER_TASK"] = str(workers_per_simulation)

    # Prevent BLAS/OpenMP oversubscription.
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = [
        "conda", "run", "-n", conda_env,
        "python", "pygad_module_parallel.py",
    ]

    log_path = sim_dir / "local_run.log"
    err_path = sim_dir / "local_run.err"

    with open(log_path, "w") as log_file, open(err_path, "w") as err_file:
        result = subprocess.run(
            cmd,
            cwd=sim_dir,
            env=env,
            stdout=log_file,
            stderr=err_file,
            text=True,
        )

    return {
        "simulation": sim_dir.name,
        "returncode": result.returncode,
        "log": str(log_path),
        "err": str(err_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run MOGA-Phonons experiment simulations locally."
    )

    parser.add_argument(
        "experiment_file",
        help="Experiment file name, e.g. experiment000007",
    )

    parser.add_argument(
        "--workers-per-simulation",
        type=int,
        default=6,
        help="Worker processes used inside each PyGAD simulation.",
    )

    parser.add_argument(
        "--simulations-in-parallel",
        type=int,
        default=1,
        help="Number of simulation directories to run at the same time.",
    )

    parser.add_argument(
        "--conda-env",
        default="moga",
        help="Conda environment name.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    experiment_path = repo_root / "experiments" / args.experiment_file
    simulations_path = repo_root / "simulations"

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    simulation_ids = read_simulation_ids(experiment_path)

    if not simulation_ids:
        raise RuntimeError(f"No simulations found in {experiment_path}")

    total_workers = args.simulations_in_parallel * args.workers_per_simulation

    print(f"Experiment: {args.experiment_file}")
    print(f"Number of simulations: {len(simulation_ids)}")
    print(f"Simulations in parallel: {args.simulations_in_parallel}")
    print(f"Workers per simulation: {args.workers_per_simulation}")
    print(f"Total requested worker processes: {total_workers}")
    print()

    if total_workers > os.cpu_count():
        print(
            f"WARNING: requested {total_workers} worker processes, "
            f"but machine reports {os.cpu_count()} CPUs."
        )
        print("This may oversubscribe the machine.\n")

    failed = []

    with ProcessPoolExecutor(max_workers=args.simulations_in_parallel) as executor:
        futures = []

        for sim_id in simulation_ids:
            sim_dir = simulations_path / sim_id

            if not sim_dir.exists():
                print(f"Skipping missing simulation directory: {sim_dir}")
                continue

            futures.append(
                executor.submit(
                    run_one_simulation,
                    sim_dir,
                    args.conda_env,
                    args.workers_per_simulation,
                )
            )

        for future in as_completed(futures):
            result = future.result()

            sim = result["simulation"]
            returncode = result["returncode"]

            if returncode == 0:
                print(f"Finished {sim}")
            else:
                print(f"FAILED {sim} with return code {returncode}")
                print(f"  stdout: {result['log']}")
                print(f"  stderr: {result['err']}")
                failed.append(sim)

    print()
    print("Local experiment run complete.")

    if failed:
        print("Failed simulations:")
        for sim in failed:
            print(f"  {sim}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
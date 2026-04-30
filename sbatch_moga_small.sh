#!/bin/bash
#SBATCH --job-name=moga
#SBATCH --account=jakar_general
#SBATCH --qos=jakar_medium_general
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:03:00
#SBATCH --output=%j.log
#SBATCH --error=%j.err

echo "========================================"
echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

ulimit -s unlimited

# Load MPI (important: must match mpi4py build)
module load gnu12
module load openmpi4/4.1.5

# Activate conda
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moga

# Diagnostics (very useful if something breaks later)
which python
python --version
which mpirun

python -c "import numpy, mpi4py; print('Env OK')"

# Go to your code
cd /home/jamunoz/MOGA-Phonons

echo "Running MOGA..."

mpirun -np $SLURM_NTASKS python pygad_module_2_6_50_mod_parallel.py > output.txt

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"

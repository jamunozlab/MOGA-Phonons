#!/bin/bash
#SBATCH --job-name=moga
#SBATCH --account=jakar_partner_physj
#SBATCH --qos=jakar_medium_physj
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=%j.log
#SBATCH --error=%j.err

echo "========================================"
echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "========================================"

ulimit -s unlimited

# Load MPI
module load gnu12
module load openmpi4/4.1.5

# Activate conda
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate moga

# Diagnostics
which python
python --version
which mpirun
python -c "import numpy, mpi4py; print(\"Env OK\")"

# Go to your code
cd /home/jamunoz/MOGA-Phonons/simulations/simulation000312/

echo "Running MOGA..."

python pygad_module_parallel.py > output.txt

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"

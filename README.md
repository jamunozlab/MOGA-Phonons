1. Install Anaconda or Miniconda (with Conda Forge)
2. Create a clean environment with Python 3.10 (recommended)
3. Install phonopy and other requirements from Conda Forge
4. Install pyGAD using PIP

To run locally (no Slurm)
5. Create directory structure with all necessary files
% python new_experiment.py genetic_phonon -t

6. Run genetic algorithm for each simulation in an experiment
% python run_experiment_local.py experiment000011 \
  --workers-per-simulation 4 \
  --simulations-in-parallel 1 \
  --conda-env moga

7. Generate dispersion yalm files
% python dispersion_generator_from_simulation_jsonl_multiprocess.py \
  --experiment experiment000011 \
  --selection top-k \
  --top-k 5 \
  --sort-by fitness_norm \
  --nproc 8 \
  --progress

8. Put the whole dataset in a pickled pandas dataframe
% python build_experiment_dataframe_updated.py \
  --experiment experiment000011 \
  --write_summary


Run pygad_module_2_6_50.py, change directory
In directory, needed files are POSCAR and band.conf, the code will generate FORCE_CONSTANTS (for each solution, replaced each time) and generation_output.txt
Run dispersion_generator_2_6_50.py to visualize

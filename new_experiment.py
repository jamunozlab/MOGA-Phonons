#!/usr/bin/env python

"""
This script is responsible for iterating over the variables that will be modified in an experiment,
creating a directory for each combination of interest with all the files necessary to run an individual
simulation in that directory, and submitting the job to the Slurm worload manager.

The user must provide code to modify the variables of interest in sim_params_dict and sbatch_options_dict,
as well as a description of the experiment. The function new_simulation_from_default should be called
once for each simulation. The script should be executed from the command line. 
"""

import argparse

from utils.new_simulation_from_default import new_simulation_from_default
from utils.dir_structure_utils import find_next_available_file
from utils.config import simulations_path, executables_path, executable, experiments_path, other_files
import utils.default_dictionaries as default_dictionaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='new_experiment',
                    description='Create new experiment. User must provide the name of a dictionary' +
                                    'in default_dictionaries.py with input parameters and default values.',
                    epilog='')
    
    parser.add_argument('default_input_dictionary_name') # positional argument
    parser.add_argument('-t', '--test', action='store_false')  # on/off flag
    
    submit_job=False

    args = parser.parse_args()
    submit_job = args.test
    default_input_dictionary = getattr(
        default_dictionaries,
        args.default_input_dictionary_name
    )
    
    simulation_ids = []


#### SCRIPT SHOULD BE BELOW ####      
    
    experiment_description = (
        "MOGA Jakar sweep. ntasks=4 \n"
        "atomic_masses: 50–100 step 10\n"
        "a_val: 2.0–3.2 step 0.2\n"
    )
    
    count = 0

    # Mass: 50 → 100 (inclusive), step 10
    masses = list(range(50, 101, 10))

    # a_val: 2.0 → 3.2 (inclusive), step 0.2
    a_vals = [round(2.0 + 0.2*i, 2) for i in range(int((3.2 - 2.0)/0.2) + 1)]

    for mass in masses:
        for a_val in a_vals:

            sim_params_dict = {
                "atomic_masses": [float(mass)],
                "a_val": float(a_val)
            }

            sbatch_options_dict = {}  # use defaults

            simulation_id = new_simulation_from_default(
                default_input_dictionary,
                simulations_path,
                executables_path,
                executable,
                sim_params_dict,
                sbatch_options_dict,
                other_files=other_files,
                submit_job=submit_job
            )

            simulation_ids.append(simulation_id)
            count += 1
        
    experiment_id = find_next_available_file('experiment', experiments_path)

#### SCRIPT SHOULD BE ABOVE ####  

    lines = experiment_description + str('\n')
    for simulation_id in simulation_ids:
        lines = lines + simulation_id + str('\n')

    with open(experiments_path + experiment_id, "w") as f:
        f.writelines(lines)
    
    print(experiment_id)
    print(simulation_ids)
    

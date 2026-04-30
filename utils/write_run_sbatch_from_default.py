#!/usr/bin/env python

"""
Auxiliary function to write sbatch files
"""

def generate_run_sbatch_lines(simulation_path, executable, options_dict):
    lines = []
    lines.append("#!/bin/bash" + str('\n'))
    for key, value in options_dict.items():
        lines.append("#SBATCH " + str(key) + " " + value + str('\n'))
    lines.append("export SLURM_CPU_BIND=\"cores\"" + str('\n'))
    first  = "srun " + simulation_path + executable + " "
    lines.append(first + 'inputc' + " > " + 'output.txt')
    return lines

def generate_run_sbatch_from_default(**kwargs):
    slurm_options_dictionary = { '-A' : 'm3766_g',
                          '-C' : 'gpu',
                          '-q' : 'shared',
                          '-t' : '06:00:00',
                          '-c' : '32',
             '--gpus-per-task=1' : '',
  #         '--ntasks-per-node' : '4',
                          '-n' : '1'
                         }
    for key, value in kwargs.items():
        slurm_options_dictionary[key] = value 
    return slurm_options_dictionary
    
def write_run_sbatch_from_default(simulation_path, executable, **kwargs):
    slurm_options_dict = generate_run_sbatch_from_default(**kwargs)
    run_sbatch_lines = generate_run_sbatch_lines(simulation_path, executable, options_dict=slurm_options_dict)
    f = open(simulation_path+"runc.sbatch", "w")
    f.writelines(run_sbatch_lines)
    f.close()
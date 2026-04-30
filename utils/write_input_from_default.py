#!/usr/bin/env python

"""
Auxiliary functions to write ferroX input files
"""

def generate_input_from_default(default, **kwargs):
    simulation_parameters_dictionary = default
    for key, value in kwargs.items():
        simulation_parameters_dictionary[key] = value 
        
    return simulation_parameters_dictionary

def generate_input_lines(param_dict):
    lines = []
    for key in param_dict.keys():
        first_part = str(key) + " = "
        value = param_dict[key]
        if isinstance(value, str):
            second_part = str(value)
        if isinstance(value, float):
            second_part = "{:.2e}".format(value)
        if isinstance(value, list):
            if isinstance(value[0], int):
                second_part = "{} {} {}".format(value[0], value[1], value[2])
            if isinstance(value[0], float):
                second_part = "{:.2e} {:.2e} {:.2e}".format(value[0], value[1], value[2])
        if isinstance(value, int):
            second_part = str(value)
        lines.append(first_part + second_part + str('\n'))
    return lines

def write_input_from_default(default, simulation_path, **kwargs):
    simulation_parameters_dictionary = generate_input_from_default(default, **kwargs)
    input_lines = generate_input_lines(param_dict=simulation_parameters_dictionary)
    f = open(simulation_path+"inputc", "w")
    f.writelines(input_lines)
    f.close()
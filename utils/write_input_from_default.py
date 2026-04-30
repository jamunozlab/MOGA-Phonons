#!/usr/bin/env python

"""
Auxiliary functions to write input files
"""

def generate_input_from_default(default, **kwargs):
    simulation_parameters_dictionary = default.copy()
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
            if all(isinstance(x, int) for x in value):
                second_part = " ".join(str(x) for x in value)
            elif all(isinstance(x, float) for x in value):
                second_part = " ".join("{:.2e}".format(x) for x in value)
            else:
                second_part = " ".join(str(x) for x in value)
        if isinstance(value, int):
            second_part = str(value)
        lines.append(first_part + second_part + str('\n'))
    return lines

def write_bcc_poscar(simulation_path, a_val, system_name="Fe"):
    lines = [
        f"For VASP. System= {system_name}\n",
        f"{a_val:.16f}\n",
        "1.0000000000000000 0.0000000000000000 0.0000000000000000\n",
        "0.0000000000000000 1.0000000000000000 0.0000000000000000\n",
        "0.0000000000000000 0.0000000000000000 1.0000000000000000\n",
        "2\n",
        "Direct\n",
        "0.0000000000000000 0.0000000000000000 0.0000000000000000\n",
        "0.5000000000000000 0.5000000000000000 0.5000000000000000\n",
    ]

    with open(simulation_path + "POSCAR", "w") as f:
        f.writelines(lines)

def write_input_from_default(default, simulation_path, **kwargs):
    simulation_parameters_dictionary = generate_input_from_default(default, **kwargs)
    input_lines = generate_input_lines(param_dict=simulation_parameters_dictionary)

    with open(simulation_path + "inputc", "w") as f:
        f.writelines(input_lines)

    if "a_val" in simulation_parameters_dictionary:
        write_bcc_poscar(
            simulation_path,
            simulation_parameters_dictionary["a_val"]
        )
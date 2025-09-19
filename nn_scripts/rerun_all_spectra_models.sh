#!/home/cschmidt/miniconda3/envs/excited_states_NN_kgcnn/bin/python

import os
from os.path import join, isdir
import subprocess
import argparse

# Parse arguments
# parser = argparse.ArgumentParser(description='Rerun all models')
# parser.add_argument('-m', '--models_folder_path', type=str, help='Path to the folder containing all models', required=True)

models_folder_path = '/data/cschmidt/excited_states/energy_NNs/fr0_project'
data_path = '/data/cschmidt/excited_states/apply_energy_NNs/fr0_project/simulated_on_s1_charges/data_from_all_sol_correct_energy/nn_data'

best_model_name = 'best_model'

# List all models
# all_models_path = [join(models_folder_path, f) for f in os.listdir(models_folder_path) if isdir(join(models_folder_path, f))]

# Only use specific models
models_to_use = [

    # 'trained_all_10',
    'train_all_10_correct_water'
    
]
all_models_path = [join(models_folder_path, f) for f in models_to_use if isdir(join(models_folder_path, f))]

# if one of those folders is called "old" remove it from the list
all_models_path = [f for f in all_models_path if 'old' not in f]

# # Execute the rerun command for all models
current_path = os.getcwd()

solvents = [file.split('_')[2].split('.')[0] for file in os.listdir(data_path) if file.endswith('.dat') and file.startswith('nn_input_')]
print(f"Found solvents: {solvents}")

for model_path in all_models_path:
    # Create results folder if it does not exist
    results_folder = join(current_path, model_path.split('/')[-1])
    if not isdir(results_folder):
        os.makedirs(results_folder)
        print(f"Created results folder: {results_folder}")

    print(model_path)
    for solvent in solvents:
        # Change to the results_folder directory
        print(f"Changing to results folder: {results_folder}")
        os.chdir(results_folder)

        # Rerun the model
        print(f"Rerunning model {join(model_path, best_model_name)} for solvent {solvent}")
        command = f"rerun_nn_spectrum.sh -f {join(data_path, f'nn_input_{solvent}.dat')} -m {join(model_path, best_model_name)} -sp {solvent}"
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
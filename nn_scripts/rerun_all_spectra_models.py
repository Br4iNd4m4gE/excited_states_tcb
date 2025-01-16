#!/home/cschmidt/miniconda3/envs/excited_states_NN_kgcnn/bin/python

import os
from os.path import join, isdir
import subprocess
import argparse

# Parse arguments
# parser = argparse.ArgumentParser(description='Rerun all models')
# parser.add_argument('-m', '--models_folder_path', type=str, help='Path to the folder containing all models', required=True)

models_folder_path = '/data/cschmidt/excited_states/energy_NNs/fr0_project'

# List all models
# all_models_path = [join(models_folder_path, f) for f in os.listdir(models_folder_path) if isdir(join(models_folder_path, f))]

# Only use specific models
models_to_use = [
    # 'train_all_10_and_traj_wrong_energy',
    # 'train_only_traj_wrong_energy',
    # 'train_only_wrong_water',
    # 'waterxNMA_only',
    # 'train_only_wrong_water',

    'trained_all_10',
    # 'train_all_10_and_traj_correct_energy',
    # 'train_all_10_and_traj_correct_energy_and_wrong_energy',
    # 'train_only_traj_wrong_and_correct_energy',
    # 'train_only_traj_correct_energy',
    # 'water_and_waterxNMA',
    # 'train_only_traj_correct_water',
    # 'train_only_traj_correct_dcm',
    # 'train_only_traj_correct_dioxane',
    # 'train_only_traj_correct_ethylacetate',
    # 'train_only_traj_correct_meoh',
    # 'train_only_traj_correct_acn',
    # 'train_only_traj_correct_hexane',
    # 'train_only_traj_correct_acetone',
    # 'train_only_traj_correct_dmf',
    # 'train_only_traj_correct_dmso'
]
all_models_path = [join(models_folder_path, f) for f in models_to_use if isdir(join(models_folder_path, f))]

# if one of those folders is called "old" remove it from the list
all_models_path = [f for f in all_models_path if 'old' not in f]

# # Execute the rerun command for all models
# current_path = os.getcwd()

for model_path in all_models_path:
    print(model_path)
    command = f"rerun_create_all_spectra.sh {model_path}"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
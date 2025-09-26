#!/home/cschmidt/miniconda3/envs/excited_states_NN_kgcnn/bin/python

import os
from os.path import join, isdir
import subprocess
import argparse

# Parse arguments
# parser = argparse.ArgumentParser(description='Rerun all models')
# parser.add_argument('-m', '--models_folder_path', type=str, help='Path to the folder containing all models', required=True)

models_folder_path = '/data/cschmidt/excited_states/energy_NNs/fr0_project'

# # Execute the rerun command for all models
# current_path = os.getcwd()
current_path = '/data/cschmidt/excited_states/apply_energy_NNs/fr0_project/simulated_on_s2_charges/model_all_10_sol_correct_energies_small_timesteps'

# data_path = '/data/cschmidt/excited_states/apply_energy_NNs/fr0_project/simulated_on_s1_charges/data_from_all_sol_correct_energy/nn_data' # s1 charges
# data_path = '/data/cschmidt/excited_states/apply_energy_NNs/fr0_project/simulated_on_s2_charges/recalc_all_geoms_from_traj/nn_data' # s2 charges
# data_path = '/data/cschmidt/excited_states/apply_energy_NNs/fr0_project/simulated_on_s2_charges/model_all_10_sol_correct_energies/nn_data' # s2 charges with model (water 100% old new model)
# data_path = '/data/cschmidt/excited_states/apply_energy_NNs/fr0_project/simulated_on_s2_charges/model_all_10_sol_correct_energies_small_timesteps/nn_data' # s2 charges high res

data_path = join(current_path, 'nn_data') # nn_data folder in current directory

best_model_name = 'best_model'

# List all models
# all_models_path = [join(models_folder_path, f) for f in os.listdir(models_folder_path) if isdir(join(models_folder_path, f))]

# Only use specific models
models_to_use = [

    'train_all_10_correct_water',
    # 'trained_all_10_old_new_water'
    
]
all_models_path = [join(models_folder_path, f) for f in models_to_use if isdir(join(models_folder_path, f))]

# Print found models
print(f"Found {len(all_models_path)} models:")
for model_path in all_models_path:
    print(f"- {model_path}")

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

        # # Rerun the model
        # print(f"Rerunning model {join(model_path, best_model_name)} for solvent {solvent}")
        # command = f"rerun_nn_spectrum.sh -f {join(data_path, f'nn_input_{solvent}.dat')} -m {join(model_path, best_model_name)} -sp {solvent}"
        # subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Rerun the model
        print(f"Rerunning model {join(model_path, best_model_name)} for solvent {solvent}")
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        command = f"/home/cschmidt/bin_public/excited_states_networks/nn_scripts/rerun_nn_spectrum.sh -g {gpu_id} -f {join(data_path, f'nn_input_{solvent}.dat')} -m {join(model_path, best_model_name)} -sp {solvent}"
        result = subprocess.run(command, shell=True, stdout=None, stderr=None)
        print(f"Command finished with return code: {result.returncode}")
#!/home/cschmidt/miniconda3/envs/excited_states_NN_kgcnn/bin/python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from os.path import join, isdir, isfile, dirname, abspath
import os
# sys.path.append("/home/cschmidt/bin/excited_states_networks") # pyNNsMD
sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import parse_single_file, extract_number_of_atoms, get_file_length, gaussian, unit_conversions, load_data_excited_states_forces
from pyNNsMD.nn_pes_src.device import set_gpu
import argparse
#from scipy.spatial.distance import pdist
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import tensorflow.keras.backend as K

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int, required=True, help="GPU ID to use")
# ap.add_argument("-p", "--outname") # wird ggf. ignoriert
ap.add_argument("-f", "--file", required=True, help="Path to the input file")
ap.add_argument("-m", "--model", required=True, help="Path to the saved model")
# ap.add_argument("-s", "--save", action="store_true", help="Save energy and oscillator strength in separate files", default=True)
args = ap.parse_args()

# Set GPU
set_gpu([args.gpuid])          ###############  wichtig !!

###############################################################

# Load unit conversions
A2Bohr, EhtoeV, ehtonm = unit_conversions["A2Bohr"], unit_conversions["EhtoeV"], unit_conversions["ehtonm"]

# Load data
inputfile = args.file
lines_to_skip = 1 # comment lines
x, y = load_data_excited_states_forces(inputfile, lines_to_skip)

# Convert to tensors
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

# Load model
model_path = args.model
best_model = tf.keras.models.load_model(model_path)

# Cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
r2_scores_energy = []
mae_scores_energy = []
r2_scores_forces = []
mae_scores_forces = []

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Scale the output data
    scaler = StandardScaler(with_std=False)
    scaler.fit(y_train)
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Evaluate the model
    test_pred_scaled = best_model.predict(x_test)
    test_pred_rescaled = scaler.inverse_transform(test_pred_scaled)

    # Calculate performance metrics
    forces_pred = K.flatten(test_pred_rescaled[:, 1:])
    forces_test = K.flatten(y_test[:, 1:])
    r2_energy = r2_score(y_test[:, 0], test_pred_rescaled[:, 0])
    mae_energy = mean_absolute_error(y_test[:, 0], test_pred_rescaled[:, 0])
    r2_forces = r2_score(forces_test, forces_pred)
    mae_forces = mean_absolute_error(forces_test, forces_pred)

    r2_scores_energy.append(r2_energy)
    mae_scores_energy.append(mae_energy)
    r2_scores_forces.append(r2_forces)
    mae_scores_forces.append(mae_forces)

# Print cross-validation results
print("Cross-Validation Results:")
print("R2 Total Energy:", np.mean(r2_scores_energy), "+/-", np.std(r2_scores_energy))
print("MAE Total Energy:", np.mean(mae_scores_energy), "+/-", np.std(mae_scores_energy), "eV")
print("R2 Forces:", np.mean(r2_scores_forces), "+/-", np.std(r2_scores_forces))
print("MAE Forces:", np.mean(mae_scores_forces), "+/-", np.std(mae_scores_forces), "eV/A")
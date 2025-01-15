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
from pyNNsMD.utils.loss import custom_loss_forces
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
import tensorflow.keras.backend as K
import joblib

ap = argparse.ArgumentParser()
# ap.add_argument("-g", "--gpuid", type=int, required=True, help="GPU ID to use")
ap.add_argument("-g", "--gpuid", type=int)
ap.add_argument("-f", "--file", required=True, help="Path to the input file")
ap.add_argument("-m", "--model", required=True, help="Path to the saved model")
ap.add_argument("-s", "--save", action="store_true", help="Save energy and oscillator strength in separate files", default=True)
args = ap.parse_args()

# Set GPU
set_gpu([args.gpuid])          ###############  wichtig !!

###############################################################

# Load unit conversions
A2Bohr, EhtoeV, ehtonm = unit_conversions["A2Bohr"], unit_conversions["EhtoeV"], unit_conversions["ehtonm"]

# Load model
model_path = args.model
with tf.keras.utils.custom_object_scope({'my_loss_fn': custom_loss_forces(0.01)}):  # Adjust the loss_ratio as needed
    best_model = tf.keras.models.load_model(model_path)

# Load data
inputfile = args.file
lines_to_skip = 1 # comment lines
x, y = load_data_excited_states_forces(inputfile, lines_to_skip)

# Convert to tensors
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

# Load scaler
scaler_folder = dirname(dirname(model_path))
scaler_path = join(scaler_folder, 'scaler.pkl')
scaler = joblib.load(scaler_path)

# Scale the output data using the loaded scaler
y_scaled = scaler.transform(y)

# Evaluate the model
pred_scaled = best_model.predict(x)
pred_rescaled = scaler.inverse_transform(pred_scaled)

# Calculate performance metrics
forces_pred = K.flatten(pred_rescaled[:, 1:])
forces_true = K.flatten(y[:, 1:])
r2_energy = r2_score(y[:, 0], pred_rescaled[:, 0])
mae_energy = mean_absolute_error(y[:, 0], pred_rescaled[:, 0])
r2_forces = r2_score(forces_true, forces_pred)
mae_forces = mean_absolute_error(forces_true, forces_pred)

# Print performance metrics
print("Performance on new data:")
print("R2 Total Energy:", r2_energy)
print("MAE Total Energy:", mae_energy, "eV")
print("R2 Forces:", r2_forces)
print("MAE Forces:", mae_forces, "eV/A")
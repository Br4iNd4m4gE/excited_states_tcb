#!/home/cschmidt/miniconda3/envs/excited_states_NN_kgcnn/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import joblib
mpl.use("Agg")

from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from os.path import join, isdir, isfile, dirname, abspath

sys.path.append(abspath(join(dirname(__file__), "..")))
from exsNN.utils.general import parse_single_file, extract_number_of_atoms, get_file_length, gaussian, unit_conversions, load_data_excited_states_forces
from exsNN.nn_pes_src.device import set_gpu
from exsNN.utils.loss import custom_loss_forces

ap = argparse.ArgumentParser()
# ap.add_argument("-g", "--gpuid", type=int, required=True, help="GPU ID to use")
ap.add_argument("-g", "--gpuid", type=int)
ap.add_argument("-f", "--file", required=True, help="Path to the input file")
ap.add_argument("-m", "--model", required=True, help="Path to the saved model")
ap.add_argument("-se", "--save_e", required=True, help="Save energy and oscillator strength in separate files", default="save_energy.txt")
ap.add_argument("-sf", "--save_f", required=True, help="Save forces and oscillator strength in separate files", default="save_forces.txt")
# ap.add_argument("-o", "--output", required=True, help="Output file for the predictions")
ap.add_argument("-l", "--loss_ratio", required=False, type=float, help="Loss ratio for the optimizer", default=0.001)
args = ap.parse_args()

# Set GPU
set_gpu([args.gpuid])          ###############  wichtig !!

###############################################################

# Load unit conversions
A2Bohr, EhtoeV, ehtonm = unit_conversions["A2Bohr"], unit_conversions["EhtoeV"], unit_conversions["ehtonm"]

# Load model
model_path = args.model
loss_ratio = args.loss_ratio
with tf.keras.utils.custom_object_scope({'my_loss_fn': custom_loss_forces(args.loss_ratio)}):  # Adjust the loss_ratio as needed
    mlmm_model = tf.keras.models.load_model(model_path, compile=False)

# Load data
inputfile = args.file
lines_to_skip = 1 # comment lines
x, y, _, _ = load_data_excited_states_forces(inputfile, lines_to_skip)

print("Shape of x:", np.array(x).shape)
print("Shape of y:", np.array(y).shape)

# Convert to tensors
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

# Evaluate the model
prediction = mlmm_model(x)

# DEBUGGING
print("Shape of y:", np.array(y).shape)
print("Shape of prediction ", prediction.shape)

# Ensure the predictions have the same shape as the original data
if prediction.shape[1] != y.shape[1]:
    raise ValueError(f"Shape mismatch: predictions have shape {prediction.shape} but expected shape {y.shape}")

# Calculate performance metrics
forces_pred = K.flatten(prediction[:, 1:])
forces_true = K.flatten(y[:, 1:])
r2_energy = r2_score(y[:, 0], prediction[:, 0])
mae_energy = mean_absolute_error(y[:, 0], prediction[:, 0])
r2_forces = r2_score(forces_true, forces_pred)
mae_forces = mean_absolute_error(forces_true, forces_pred)

# Print performance metrics
print("Performance on new data:")
print("R2 Total Energy:", r2_energy)
print("MAE Total Energy:", mae_energy, "eV")
print("R2 Forces:", r2_forces)
print("MAE Forces:", mae_forces, "eV/A")

# Save the predictions
if args.save_f:
    output_file = args.save_f
    with open(output_file, "w") as f:
        f.write("Forces [eV/A]\n")
        for i in range(len(forces_pred)):
            f.write(f"{forces_pred[i]}\n")

    print(f"Force predictions saved to {output_file}")

if args.save_e:
    output_file = args.save_e
    with open(output_file, "w") as f:
        f.write("Total Energy [eV]\n")
        for i in range(len(prediction[:, 0])):
            f.write(f"{prediction[i, 0]}\n")

    print(f"Energy predictions saved to {output_file}")

## Plot predictions vs true values
# Energy
plt.figure()
plt.hist2d(y[:, 0], prediction[:, 0], bins=100, cmin=1, cmap='inferno', label="Total Energy")
plt.colorbar()
plt.xlabel("True Total Energy [eV]")
plt.ylabel("Predicted Total Energy [eV]")
plt.title("Total Energy Predictions")
plt.savefig("total_energy_predictions.png")

# Forces cmin=1
plt.figure()
plt.hist2d(forces_true, forces_pred, bins=100, cmin=1, cmap='inferno', label="Forces")
plt
plt.xlabel("True Forces [eV/A]")
plt.ylabel("Predicted Forces [eV/A]")
plt.title("Forces Predictions")
plt.savefig("forces_predictions_cmin1.png")

# Forces cmin=50
plt.figure()
plt.hist2d(forces_true, forces_pred, bins=100, cmin=50, cmap='inferno', label="Forces")
plt
plt.xlabel("True Forces [eV/A]")
plt.ylabel("Predicted Forces [eV/A]")
plt.title("Forces Predictions")
plt.savefig("forces_predictions_cmin50.png")
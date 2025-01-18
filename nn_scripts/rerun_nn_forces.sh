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
ap.add_argument("-o", "--output", required=True, help="Output file for the predictions")
args = ap.parse_args()

# Set GPU
set_gpu([args.gpuid])          ###############  wichtig !!

###############################################################

# Load unit conversions
A2Bohr, EhtoeV, ehtonm = unit_conversions["A2Bohr"], unit_conversions["EhtoeV"], unit_conversions["ehtonm"]

# Load model
model_path = args.model
with tf.keras.utils.custom_object_scope({'my_loss_fn': custom_loss_forces(0.005)}):  # Adjust the loss_ratio as needed
    mlmm_model = tf.keras.models.load_model(model_path)

# Load data
inputfile = args.file
lines_to_skip = 1 # comment lines
x, y, _, _ = load_data_excited_states_forces(inputfile, lines_to_skip)

print("Shape of x:", np.array(x).shape)
print("Shape of y:", np.array(y).shape)

# Convert to tensors
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

# # Load scaler
# scaler_folder = dirname(dirname(model_path))
# scaler_path = join(scaler_folder, 'scaler.pkl')
# scaler = joblib.load(scaler_path)

# # Scale the output data using the loaded scaler
# y = scaler.transform(y)

print("Shape of y:", np.array(y).shape)

# Evaluate the model
# prediction = mlmm_model.predict(x)
prediction = mlmm_model(x)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(prediction.shape)
# print(prediction[0])
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

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

if args.save:
    # Save the predictions
    output_file = args.output
    with open(output_file, "w") as f:
        f.write("Total Energy [eV]  Forces [eV/A]\n")
        for i, pred in enumerate(prediction):
            pred_str = " ".join(map(str, pred.numpy().flatten()))
            f.write(f"{pred_str}\n")

    print(f"Predictions saved to {output_file}")

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
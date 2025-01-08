#!/home/cschmidt/miniconda3/envs/excited_states_NN_from_monja/bin/python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from os.path import join, isdir, isfile, dirname, abspath
import os
# sys.path.append("/home/cschmidt/bin/excited_states_networks") # pyNNsMD
sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import parse_single_file, extract_number_of_atoms, get_file_length, gaussian, unit_conversions
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

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
# ap.add_argument("-p", "--outname") # wird ggf. ignoriert
ap.add_argument("-f", "--file", required=True, help="Path to the input file")
ap.add_argument("-m", "--model", required=False, help="Path to the model", default=None)
ap.add_argument("-s", "--save", action="store_true", help="Save energy and oscillator strength in separate files", default=True)
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

###################### Define Functions ######################

import numpy as np
from sklearn.model_selection import KFold

# Number of folds for cross-validation
n_folds = 3

# Initialize KFold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays to store results
r2_scores_energy = []
mae_scores_energy = []
r2_scores_forces = []
mae_scores_forces = []

# Perform cross-validation
for train_index, test_index in kf.split(x):
    # Split the data
    x_train, x_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    # Convert to tensors
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    # Scale the output data
    scaler.fit(y_train)
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Hyperparameter tuning
    tuner.search(x_train, y_train_scaled, batch_size=batch_size, epochs=hp_epochs, callbacks=[stop_early], verbose=2, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # Train the best model
    best_model.fit(x_train, y_train_scaled, batch_size=batch_size, epochs=fit_epochs, verbose=2, validation_split=0.2)

    # Evaluate the best model
    test_pred_scaled = best_model.predict(x_test)
    test_pred_rescaled = scaler.inverse_transform(test_pred_scaled)

    # Calculate performance metrics
    forces_pred = K.flatten(test_pred_rescaled[:, 1:])
    forces_test = K.flatten(y_test[:, 1:])
    r2_energy = r2_score(y_test[:, 0], test_pred_rescaled[:, 0])
    mae_energy = mean_absolute_error(y_test[:, 0], test_pred_rescaled[:, 0])
    r2_forces = r2_score(forces_test, forces_pred)
    mae_forces = mean_absolute_error(forces_test, forces_pred)

    # Store results
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
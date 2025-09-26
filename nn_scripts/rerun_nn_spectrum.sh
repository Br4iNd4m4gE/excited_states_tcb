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
mpl.use("Agg")

from os.path import join, isdir, isfile, dirname, abspath
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit

sys.path.append(abspath(join(dirname(__file__), "..")))
from exsNN.utils.general import parse_single_file, extract_number_of_atoms, get_file_length, gaussian, unit_conversions
from exsNN.nn_pes_src.device import set_gpu

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
ap.add_argument("-f", "--file", required=True, help="Path to the input file")
ap.add_argument("-m", "--model", required=True, help="Path to the model", default=None)
ap.add_argument("-s", "--save", action="store_true", help="Save energy and oscillator strength in separate files", default=True)
ap.add_argument("-sp", "--save_prefix", help="Prefix for save file.", default="")
ap.add_argument("-l", "--lines", type=int, help="Number of (comment) lines to skip in the input file", default=0)
ap.add_argument("-b", "--batch_size", type=int, default=1000, help="Batch size for prediction")
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

###############################################################

keep_energy_and_osc = args.save # you want energy and osc. str. saved in separate files
parent_path = os.getcwd() # path for evaluation
data_path = args.file
lines_to_skip = args.lines # != 0 in case of comment lines in data (e.g. energy line)
batch_size = args.batch_size

## Load model
model_path = args.model
print(f"Model Path: {abspath(model_path)}")

# Load model (iput: coords in Bohr and ESP in Hartree; output: energy in Hartree and oscillator strength)
model = tf.saved_model.load(model_path)

# Load unit conversions
A2Bohr, EhtoeV = unit_conversions["A2Bohr"], unit_conversions["EhtoeV"]

# Load Stuff
natoms = extract_number_of_atoms(data_path, lines_to_skip)
print(f"Number of atoms: {natoms} within a single molecule.")

# Determine size of the data used for all solvents, by searching for the smallest file
linestotal = get_file_length(data_path)
ntotal = linestotal / (natoms + lines_to_skip + 1)
if not ntotal.is_integer():
    raise ValueError("Number of Lines incorrect.")

ntotal = int(ntotal)
print(f"Total number of structures: {ntotal}")

## 1. Load data
print("loading")
xyz_data = np.zeros((ntotal, natoms, 4))

# Read xyz data and ESP
with open(data_path, "r") as data:
    for i in range(ntotal):
        for j in range(natoms + lines_to_skip):
            line = data.readline()
            if lines_to_skip != 0: # skip comment lines in data (in case of e.g. energy line)
                if j < lines_to_skip:
                    continue
            xyz_data[i, j - lines_to_skip] = [float(x) for x in line.split()[1:]]
        data.readline()
print("loaded")

xyz_data = np.asarray(xyz_data, dtype=np.float32)

# keep preprocessing 
coords = xyz_data[:,:,:3] * A2Bohr
esp    = xyz_data[:,:,3]

# Predict energy and oscillator strength in batches
print(f"Processing {ntotal} structures in batches of {batch_size}")
n_batches = (ntotal + batch_size - 1) // batch_size

predictions = []
for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, ntotal)
    
    batch_coords = coords[start_idx:end_idx]
    batch_esp = esp[start_idx:end_idx]
    x_batch = (batch_coords, batch_esp)
    
    batch_pred = model(x_batch)
    predictions.append(batch_pred)
    print(f"Processed batch {i+1}/{n_batches} (structures {start_idx}-{end_idx-1})")

# Concatenate all predictions
pred = tf.concat(predictions, axis=0)
predlist = pred.numpy()

# Convert energy to eV
pred_eV = predlist[:,0] * EhtoeV
print(f"Final prediction shape: {len(pred_eV)}")

# Store energy and osc. str. data in separate files
if keep_energy_and_osc:
    file_energy_name = f"{args.save_prefix}_nn_pred_energy.dat"
    file_osc_str_name = f"{args.save_prefix}_nn_pred_osc_str.dat"
    file_energy_osc_str_name = f"{args.save_prefix}_nn_pred_energy_osc_str.dat"

    file_energy_path = join(parent_path, file_energy_name)
    file_osc_str_path = join(parent_path, file_osc_str_name)
    file_energy_osc_str_path = join(parent_path, file_energy_osc_str_name)

    # Save both in a single file with a headline: "Energy (eV) Oscillator Strength"
    with open(file_energy_osc_str_path, "w") as f:
        f.write("# Energy (eV)    Oscillator Strength\n")
        np.savetxt(f, np.c_[pred_eV, predlist[:, 1]])

print("Prediction completed successfully!")
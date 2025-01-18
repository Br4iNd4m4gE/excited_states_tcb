#!/home/cschmidt/miniconda3/envs/excited_states_NN_kgcnn/bin/python
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
ap.add_argument("-m", "--model", required=True, help="Path to the model", default=None)
ap.add_argument("-s", "--save", action="store_true", help="Save energy and oscillator strength in separate files", default=True)
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

###############################################################

keep_energy_and_osc = args.save # you want energy and osc. str. saved in separate files
parent_path = os.getcwd() # path for evaluation
data_path = args.file
lines_to_skip = 1 # modular via argparse

## Load model
# If model name is given as argument, use this model
model_path = args.model
print(f"Model Path: {abspath(model_path)}")

# Load model (iput: coords in Bohr and ESP in Hartree; output: energy in Hartree and oscillator strength)
model = tf.saved_model.load(model_path)

# Load unit conversions
A2Bohr, EhtoeV, ehtonm = unit_conversions["A2Bohr"], unit_conversions["EhtoeV"], unit_conversions["ehtonm"]

# Load Stuff
natoms = extract_number_of_atoms(data_path, lines_to_skip)
print(f"Number of atoms: {natoms} within a single molecule.")

# Determine size of the data used for all solvents, by searching for the smallest file
linestotal = get_file_length(data_path)
ntotal = linestotal / (natoms + lines_to_skip + 1)
if not ntotal.is_integer():
    raise ValueError("Number of Lines incorrect.")

ntotal = int(ntotal)

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

x_data = (coords, esp) # for oder models [coords, esp]

# Predict energy and oscillator strength
pred = model(x_data)
predlist = pred.numpy()

# Convert energy to eV
pred_eV = predlist[:,0] * EhtoeV
print(len(pred_eV))

# Plot histogram
# plt.clf()
try:
	# Create histogram with normalization
	n, bins, patches = plt.hist(pred_eV, bins=100, weights=predlist[:, 1], alpha=0.5, density=True) #,range=[2.5,4.5])
	binwidth = bins[1] - bins[0]
	binmids = bins + 0.5 * binwidth

	# Fit Gaussian
	popt, pcov = curve_fit(gaussian, binmids[:-1], n, p0=[2, 3.5, 0.5])
	plt.plot(binmids[:-1], gaussian(binmids[:-1], *popt), color='r')
	print(f"Mean: {popt[1]}, Variance: {popt[2]}")
	plt.xlabel("Excitation Energy [eV]")
	plt.ylabel("Count [a.u.]")
	# plt.legend()

	fig_name = "rerun_histogram.png"
	fig_path = join(parent_path, fig_name)
	plt.savefig(fig_path)
except:
	print("Gaussian fit failed.")
	pass

# Store energy and osc. str. data in separate files
if keep_energy_and_osc:
	file_energy_name = "rerun_energies.txt"
	file_osc_str_name = "rerun_osc_str.txt"
	file_energy_path = join(parent_path, file_energy_name)
	file_osc_str_path = join(parent_path, file_osc_str_name)
	np.savetxt(file_energy_path, pred_eV)
	np.savetxt(file_osc_str_path, predlist[:, 1])

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
ap.add_argument("-m", "--model", required=False, help="Path to the model", default=None)
ap.add_argument("-s", "--save", action="store_true", help="Save energy and oscillator strength in separate files", default=True)
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

###################### Define Functions ######################

##### PARAMETERS #####
# model_parent_path can also be given by "-m" parser argument, then the line below is not needed
model_parent_path = '/data/cschmidt/excited_states/energy_NNs/retinal_project/retinal_beryl_delta_try2' # path in which the best_model ist
model_name = 'best_model' # name of the model
parameter_file_name = 'params.txt' # name of the file in which the parameters are stored

###############################################################

keep_energy_and_osc = args.save # you want energy and osc. str. saved in separate files
parent_path = os.getcwd() # path for evaluation
data_name = args.file
natoms = extract_number_of_atoms(join(parent_path, data_name))
print(f"Number of atoms: {natoms}")

## Load model
# If model name is given as argument, use this model
if args.model is not None:
	model_parent_path = args.model
model_path = join(model_parent_path, model_name)
print(f"Model Path: {model_path}")

model = tf.saved_model.load(model_path)

# Load model parameters
params = np.loadtxt(join(model_path, parameter_file_name), usecols=(0), max_rows=6)
#model.summary()
xmean, xstd, ymean, ystd, oscmean, oscstd = params[0:6]

# Load unit conversions
A2Bohr, EhtoeV, ehtonm = unit_conversions["A2Bohr"], unit_conversions["EhtoeV"], unit_conversions["ehtonm"]

# Define data path
data_path = join(parent_path, data_name)

# Determine size of the data used for all solvents, by searching for the smallest file
size = None
linestotal = get_file_length(data_path)
ntotal = linestotal / (natoms + 1)
if not ntotal.is_integer():
    raise ValueError("Number of Lines incorrect.")
data_size = int(ntotal)
size = data_size
 
# get number of molecules
result = subprocess.run(['wc', '-l', data_path], capture_output=True, text=True)
max_n_molecules = int(result.stdout.split()[0])
max_n_molecules /= natoms + 1

##### 1. Data extraction
# load data
print("loading")
xyz_data = np.zeros((size, natoms, 4))
# Read xyz data and ESP
with open(data_path, "r") as data:
	for i in range(size):
		if i > max_n_molecules:
			raise ValueError("Something went horribly wrong. Look in the Code!")  # if size of data is smaller than the "size" variable
		for j in range(natoms):
			line = data.readline()
			xyz_data[i, j] = [float(x) for x in line.split()[1:]]
		data.readline()
print("loaded")
#print(xyz_data[1,:,:])
xyz_data = np.asarray(xyz_data, dtype=np.float32)
# keep preprocessing 
coords = xyz_data[:,:,:3] # is (nrdata, 85, 3) or (nr, 170,3)
coords *= A2Bohr
esp_raw = xyz_data[:,:,3] # is (nrdata, 85)

# store test data in dictionary
datas = {"x": coords, "esp": esp_raw}

datas["x_scaled"] = (datas["x"] - xmean) / xstd

#data["invd"] = pdist(data["x_scaled"])
pred = model([datas["x_scaled"], datas["esp"]])
predlist = pred.numpy()
#predlist=np.ndarray.flatten(predlist)
predlist[:,0] = predlist[:,0] * ystd + ymean
predlist[:,1] = predlist[:,1] * oscstd + oscmean
#energies_nm=ehtonm/energies[10000:]
#pred_nm=ehtonm/predlist[:,0]
pred_eV = EhtoeV * predlist[:,0]
print(len(pred_eV))

# Clear figure
plt.clf()

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

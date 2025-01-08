#!/home/mkunkel/miniconda3/envs/clonednew/bin/python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
# sys.path.append("/home/cschmidt/bin/excited_states_networks") # pyNNsMD
sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import parse_single_file
from pyNNsMD.nn_pes_src.device import set_gpu
import argparse
#from scipy.spatial.distance import pdist
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
from os.path import join
import os
import subprocess

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
ap.add_argument("-p", "--outname") # wird ggf. ignoriert
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

##### PARAMETERS #####
# parent_path = "/data/cschmidt/excited_states/energy_apply_NNs/inacetone"
parent_path = os.getcwd() # path for evaluation
model_parent_path = '/data/cschmidt/excited_states/energy_NNs/trained_fr0_all_solvents_w_old_data' # path in which the best_model ist
# model_parent_path = '/data/cschmidt/excited_states/energy_NNs/test_manu_data'
model_name = 'best_model' # name of the model
parameter_file_name = 'params.txt' # name of the file in which the parameters are stored
# testdata = "/data/user6/menns/complete/es/loss4_nn_input_acn_sA.dat" # full path of testdata
data_name = 'nn_input.dat'
natoms = 45

######################

model_path = join(model_parent_path, model_name)

model = tf.saved_model.load(model_path)

params = np.loadtxt(join(model_path, parameter_file_name), usecols=(0), max_rows=6)
#model.summary()
xmean, xstd, ymean, ystd, oscmean, oscstd = params[0:6]

A2Bohr = 1.8897259886
EhtoeV = 27.2114
ehtonm = 299792458 * 6.62607015e-7 / 4.3597482
size = 3429

testdata = join(parent_path, data_name)
 
# get number of molecules
result = subprocess.run(['wc', '-l', testdata], capture_output=True, text=True)
max_n_molecules = int(result.stdout.split()[0])
max_n_molecules /= natoms + 1

##### 1. Data extraction
# load data
print("loading")
xyz_data=np.zeros((size, natoms, 4))
with open(testdata,"r") as data:
	for i in range(size):
		if i >= max_n_molecules: break # if size of data is smaller than the "size" variable
		for j in range(natoms):
			line=data.readline()
			# print(f"i ist {i}; j it {j} line ist \n{line}")
			xyz_data[i,j]=[float(x) for x in line.split()[1:]]
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

# Create histogram with normalization
n, bins, patches = plt.hist(pred_eV, bins=100, weights=predlist[:, 1], density=True) #,range=[2.5,4.5])
binwidth = bins[1] - bins[0]
binmids = bins + 0.5 * binwidth

# Fit Gaussian
popt, pcov = curve_fit(gaussian, binmids[:-1], n, p0=[2, 3.5, 0.5])
plt.plot(binmids[:-1], gaussian(binmids[:-1], *popt))
print(f"Mean: {popt[1]}, Variance: {popt[2]}")
plt.xlabel("Excitation Energy [eV]")

fig_name = "rerun_histogram.png"
fig_path = join(parent_path, fig_name)
plt.savefig(fig_path)

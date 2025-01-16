'''This script builds and trains a neural network to predict excitation/emission energies and oscillator strengths'''

import os
import sys
import numpy as np
import argparse
from os.path import join, isdir, isfile, dirname, abspath
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

import tensorflow.keras as ks
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import parse_single_file, shuffle_and_split, generate_invd_list, get_file_length, extract_number_of_atoms, unit_conversions
from pyNNsMD.nn_pes_src.device import set_gpu
from pyNNsMD.utils.loss import r2_metric
from pyNNsMD.layers.mlp import MLP
from pyNNsMD.layers.features import FeatureGeometric
from pyNNsMD.layers.normalize import ConstLayerNormalization
from pyNNsMD.models.hp import hpModelBuilder_energy_oscStr

from pyNNsMD.esp_nn import precompute_feature_in_chunks, set_const_normalization_from_features
from pyNNsMD.esp_nn import OutputSpec, SubNet, build_model, get_limits
from pyNNsMD.esp_nn import build_geom_preprocess_layer, ScaledMeanAbsoluteError
from sklearn.metrics import mean_absolute_error, r2_score

import subprocess

############################

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
# ap.add_argument("-p", "--outname") # wird ggf. ignoriert
ap.add_argument("-f", "--file")
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

###########################   User section   #################################

# paths and output
traindata = args.file
outpath = os.getcwd() # Here stuff is written
mod_outpath=join(outpath, "best_model") # for model, params, unused indices 
hp_outpath=join(outpath, "outputtuner") # save tuner trials
logfile = join(outpath, "logfile.txt") # writes all stdout into this file

# training data and scenario
lines_to_skip = 1 # number of lines to skip in the input file, not containing atom coordinates
natoms = extract_number_of_atoms(traindata, lines_to_skip)
print(f"Number of atoms detected in the input file {traindata} is {natoms}")
coords_to_atomic = True
esp_in_traindata = True
scale_esp = False # if esp_in_traindata==False: the value of scale_esp doesn't matter
linestotal = get_file_length(traindata)
ntotal = linestotal / (natoms + 2)
if not ntotal.is_integer():
    raise ValueError("Number of Lines incorrect.")
ntotal = int(ntotal)
trainPercentage = 0.9
ntrain = int(ntotal * trainPercentage)
ntest = ntotal - ntrain

# all models 
norm = "const" #  const normalizes geometries once over all data ; 'batch' in batchs
loss = "mean_squared_error" # for trainig 
final_activ = "linear" # activation function of the last layers of the MLPs 
dense_activ = {'class_name': "leaky_softplus", "config": {'alpha': 0.03}} # for MLP
epochs = 2000 # for both models, needed for model.fit()
callback_patience = 250 # how many epoches without improvement are tolerated

# hyperparameter search
hp_dict = {
    "neurons_min":    20, # only hp-model
    "neurons_max":    100, # only hp-model
    "neurons_step":   5, # only hp-model
    "layers_min":     2, # only hp-model
    "layers_max":     8, # only hp-model
    "layers_step":    1, # only hp-model
    "regulizer":      "l2", # only hp-model
    "learning_rates": [1e-3, 5e-4, 1e-4, 5e-5], # only hp-model
}

hp_maxepochs =   20 # for tuner object 
hp_factor  =    2 # for tuner object

# original-esp model to compare to
origmod_outpath = join(outpath, "orig_best_model") # for model, params, unused indices 
orig_neurons = 30 # neurons in mlp layer in orig-model
orig_learning_rate = 1e-4 # learning rate of orig-model
orig_layer_depth = 2 # depth of mlp layer for oig-model

# plotting
plot_scatters = True # plot predicted_energy(eV) vs. ref_energy (eV) 
plot_learning_curve = True # plot log(mse) vs. epochs

# other
clean_up_hpoutpath = True # delet hp_outpath before tuning (catch some errors 'oracle exited training' etc.)

########################  End of User section  ################################

# Logger so that output will be written to both terminal and stdout
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        pass  
sys.stdout=Logger()

###########################  Start of Sript  ##################################

##### 0. Constants and Definitions
Ha2eV, A2Bohr = unit_conversions["EhtoeV"], unit_conversions["A2Bohr"]

stop_early = ks.callbacks.EarlyStopping(monitor='val_loss', # which quantity to monitor
                                        patience=callback_patience, # how many epochs without improvement to tolerate
                                        restore_best_weights=True # reset weights to those of best model
                                                                  # after stopping (recommended)
                                        )
lr_reduction = ks.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=20,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-6)


##### 1. Data extraction
# load data, shuffle and split, save indizes of unused data (=test data)
xyz_data, energies = parse_single_file(traindata, natoms) # energies and osc. str.

print(f"Number of Data points in the input file {traindata} is {len(xyz_data)}")
    
# Check if data is enough for train and test
if len(xyz_data) < (ntrain+ntest):
    print("ERROR: Dataset (%i) is not large enough for the selected ntrain and ntest" %(len(xyz_data)))
    sys.exit()

#check distribution
plt.hist(energies[:,1],bins=20)
plt.savefig(join(outpath, "osc_distribution.png"), dpi=300)

# keep preprocessing 
coords = xyz_data[:, :, 1:4] # is (nrdata, 85, 3) or (nr, 170,3)
if coords_to_atomic == True:
    print("Converting coords to atomic units")
    coords *= A2Bohr
else:
    print("Assuming coords are already in atomic units!")

esp_raw = xyz_data[:, :, 4] # is (nrdata, 85)

x, esp_tmp, ene, unused = shuffle_and_split(xyz_data, energies, ntrain)
    
esp = esp_tmp[:, :, 0]
# print(x.shape, esp.shape, ene.shape) # x=(nrdata, 85, 3), esp=(nr,85), ene=(nr,)

# store train and test data in dictionary
data = {"x": x, "esp":esp}

# scale esp
if scale_esp:
    data["esp"] = StandardScaler().fit_transform(data["esp"])

# append dictionaries with energy
data["targets"] = ene

#check distribution
plt.hist(ene[:,1],bins=20)
plt.savefig(join(outpath, "osc_distribution.png"), dpi=300)

## scaling
geoscaler = StandardScaler()
data["x_mean"]   = geoscaler.fit(data["x"].reshape(-1, 1)).mean_
data["x_var"]    = geoscaler.var_
data["x_scaled"] = (data["x"] - data["x_mean"]) / (data["x_var"] ** 0.5)

# same with energy
targetscaler = StandardScaler()
data["targets_scaled"] = targetscaler.fit_transform(data["targets"].reshape(-1, 2))
data["targets_mean"]   = targetscaler.mean_
data["targets_var"]    = targetscaler.var_
target                 = data["targets_scaled"]
    
# otput_spec with scaler being already fitted to energy -> Must come after scaler.fit
output_spec = { # this is ugly, as output_spec is needed in hp_simple_model()
    "QM/MM energy" : OutputSpec(from_subnets = "monolith",
                                scaler = targetscaler
                                )                                                
} 

# x and y data
if esp_in_traindata:
    x_train = [data["x_scaled"], data["esp"]]
    y_train = data["targets_scaled"]
else:
    x_train = data["x_scaled"]
    y_train = data["targets_scaled"]

## 2. Hyperparameter Search
# this is needed if tuner quits with "INFO:tensorflow:Oracle triggered exit"
if clean_up_hpoutpath: # removes directory which can be necessary 
    import shutil 
    if os.path.isdir(hp_outpath):
        shutil.rmtree(hp_outpath) # = bash's rm -rf

# Initialize ModelBuilder
model_builder = hpModelBuilder_energy_oscStr(hp_dict, natoms, esp_in_traindata, dense_activ, final_activ, output_spec, loss, r2_metric, norm)

# Perform hyperparameter search
best_hps, tuner = model_builder.perform_hp_search(x_train, y_train, hp_maxepochs, hp_factor, stop_early, lr_reduction, hp_outpath)

print("------------------------------------------")
print(f'''{best_hps.get("neurons")} neurons, {best_hps.get("layers")} layers, {best_hps.get("loss_ratio")} loss ratio, {best_hps.get("initial_lr")} initial learning rate and {best_hps.get("l2_penalty")} regulization penalty give the best results''')
print("------------------------------------------")

# Build and train the best model
hp_model = tuner.hypermodel.build(best_hps)


##### 4. Training
# Train the Model: First, calculate the representations for all training 
# data points to set scalers to mean and std. Then fit.

# pre-calculate geometries to fit the feat_std layers for normalizing inv.dists
if norm == "const": # eigentlich immer oder?
    # precomputing features means (nrdata, 85, 3) -> (nrdata, 3570) 
    feat_precomp = precompute_feature_in_chunks(data["x_scaled"], hp_model, batch_size=32)  # changed for test
    # now the scaler of feat_std layer must be set. So weights and biases of this
    # layer must be so that x -> x-µ/std
    set_const_normalization_from_features(feat_precomp, hp_model)

# fit the models
if esp_in_traindata:
    hp_hist = hp_model.fit([data["x_scaled"], data["esp"]], target, epochs=epochs,
                     validation_split=0.1, verbose=2, callbacks=[stop_early,lr_reduction])

elif not esp_in_traindata:
    hp_hist = hp_model.fit(data["x_scaled"], target, epochs=epochs,
                     validation_split=0.1, verbose=2, callbacks=[stop_early])

# Save model
hp_model.save(mod_outpath)

# Get best epoch
hp_best_epoch = np.argmin(hp_hist.history["val_loss"])


##### 5. Prediction
# Scale testdata with mean and var from traindata
testcoords=(coords[unused[:ntest]]-data["x_mean"])/(data["x_var"]**0.5)
# Predictions
if  esp_in_traindata:
    hp_pred = hp_model.predict([testcoords,esp_raw[unused[:ntest]]])
elif not esp_in_traindata:
    hp_pred = hp_model.predict(testcoords)
    
##### 6. Evaluation
# scale the normalized prediction back to the natural scale of the data
hp_pred_scaled = targetscaler.inverse_transform(hp_pred)
# scale the references for the test data to the normalized scale for evaluation
ref_scaled = targetscaler.transform(energies[unused[:ntest]].reshape(-1,2))

# get rescaled predictions in eV
energies_eV=energies[unused[:ntest],0]*27.2114
hp_pred_scaled_eV=hp_pred_scaled[:,0]*27.2114

# run keras model evaluation
if esp_in_traindata:
    hp_metrics = hp_model.evaluate([testcoords, esp_raw[unused[:ntest]]], ref_scaled)
elif not esp_in_traindata:
    hp_metrics = hp_model.evaluate(testcoords, ref_scaled)

# Output important metrics    
print("\n\n", 22 * "-", "\n\t\tSummary\n", 22 * "-")
print("\n\tHP search lead to:\n", best_hps.get_config()["values"])

print("\n\tPerformance of HP model:")
test_mae_eV = mean_absolute_error(energies_eV,hp_pred_scaled_eV)
test_mae_osc = mean_absolute_error(energies[unused[:ntest],1],hp_pred_scaled[:,1])
print("test loss: ", hp_metrics[0])
print("test MAE (eV): ", test_mae_eV)
print("test MAE osc: ", test_mae_osc)
print("test R2: ", hp_metrics[2])
print("full metrics: ", hp_metrics)
print(f"best epoch: {hp_best_epoch}")
print("---")
print("best train loss (atomic): ", hp_hist.history["loss"][hp_best_epoch])
print("best train R2: ", hp_hist.history["r2_metric"][hp_best_epoch])
print("---")
print("best val loss (atomic): ", hp_hist.history["val_loss"][hp_best_epoch])
print("best val R2: ", hp_hist.history["val_r2_metric"][hp_best_epoch])


##### 7. Write information to files
# save indices not used for training to file for later use in tests
np.savetxt(join(mod_outpath, "indices_for_testing_all.txt"), unused)
np.savetxt(join(mod_outpath, "indices_for_testing_testdata.txt"), unused[:ntest])

#check distribution
plt.hist(energies[:,1],bins=20)
plt.savefig(join(outpath, "osc_distribution_last.png"), dpi=300)

# save energies to plot scatters
np.savetxt(join(mod_outpath, 'hp_ref_energies_eV.dat'), energies_eV)
np.savetxt(join(mod_outpath, 'hp_ref_osc.dat'), energies[unused[:ntest],1])
np.savetxt(join(mod_outpath, 'hp_predicted_energies_eV.dat'), hp_pred_scaled_eV)
np.savetxt(join(mod_outpath, 'hp_predicted_osc.dat'), hp_pred_scaled[:,1])
    
##### 8. Plotting
# Plot training progress
if plot_learning_curve:
    f, ax = plt.subplots(1, figsize=(6,6))
    ax.plot(np.arange(len(hp_hist.history["loss"])), hp_hist.history["loss"], label="training loss")
    ax.plot(np.arange(len(hp_hist.history["loss"])), hp_hist.history["val_loss"], label="validation loss")
    ax.set_yscale("log")
    ax.set_ylabel("log(MSE)")
    ax.set_xlabel("epochs")
    ax.plot(hp_best_epoch, hp_hist.history["val_loss"][hp_best_epoch], ls="", marker="x", ms=10, c="black", label="best model")
    ax.axvline(x=hp_best_epoch, color="gray", ls="--")
    ax.axhline(y=hp_hist.history["val_loss"][hp_best_epoch], color="gray", ls="--")
    ax.legend()
    f.savefig(join(mod_outpath, "test-loss.png"), dpi=300)
        
# Plot Scatter
if plot_scatters:
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.hist2d(energies_eV, hp_pred_scaled_eV.flatten(),
                  bins=1000, # Just for Tests; Mila wrote 1000
                  cmin=1, # Just for Tests; Mila wrote 1 
                  norm=mcolors.PowerNorm(0.5))
    b,t = get_limits([energies_eV, hp_pred_scaled_eV])
    ax.set_xlabel("reference (eV)")
    ax.set_ylabel("prediction (eV)")
    ax.set_aspect("equal")
    ax.set_ylim((b,t))
    opti_ref = np.linspace(b,t,num=10)
    ax.plot(opti_ref, opti_ref, c="C1")
    fig.savefig(join(mod_outpath, "scatter.png"), dpi=300)
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.hist2d(energies[unused[:ntest],1], hp_pred_scaled[:,1],
                  bins=1000, # Just for Tests; Mila wrote 1000
                  cmin=1, # Just for Tests; Mila wrote 1 
                  norm=mcolors.PowerNorm(0.5))
    b, t = get_limits([energies[unused[:ntest],1], hp_pred_scaled[:,1]])
    ax.set_xlabel("reference")
    ax.set_ylabel("prediction")
    ax.set_aspect("equal")
    ax.set_ylim((b,t))
    opti_ref = np.linspace(b,t,num=10)
    ax.plot(opti_ref, opti_ref, c="C1")
    fig.savefig(join(mod_outpath, "scatter_osc.png"), dpi=300)

# save scaling parameters to GROMACS-readable format
hypers = [
    data["x_mean"][0], 
    (data["x_var"]**0.5)[0], 
    data["targets_mean"][0], 
    (data["targets_var"]**0.5)[0], 
    data["targets_mean"][1], 
    (data["targets_var"]**0.5)[1],
    (0,0) # no gradients
    
    ]
with open(os.path.join(mod_outpath, "params.txt"), "w") as outf:
    for k in hypers:
        outf.write(f"{k}\n")

##### 9. Clean up
# if delete_tunertrials:
#     shutil.rmtree(hp_outpath) # = bash's rm -rf
    


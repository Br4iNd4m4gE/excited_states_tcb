'''This script builds and trains a neural network to predict excitation/emission energies and oscillator strengths'''

import os
import sys
import numpy as np
import argparse
import shutil
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow.keras as ks
import matplotlib as mpl
mpl.use('Agg')

from os.path import join, isdir, isfile, dirname, abspath
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import shuffle_and_split_train_test, unit_conversions, load_data_excited_states_energies, read_json_config
from pyNNsMD.nn_pes_src.device import set_gpu
from pyNNsMD.utils.loss import r2_metric
from pyNNsMD.models.hp import hpModelBuilder_energy_oscStr
from pyNNsMD.esp_nn import precompute_feature_in_chunks, set_const_normalization_from_features, OutputSpec, get_limits

############################ PARSE ARGUMENTS ##################################

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
# ap.add_argument("-p", "--outname") # wird ggf. ignoriert
ap.add_argument("-f", "--file", required=True, type=str, dest="file", action="store", help="Path to input file", metavar="file")
ap.add_argument("-c", "--conf", default=None, type=str, dest="conf", action="store", required=False, help="Path to config file, default: None", metavar="config")
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

############################ CONFIG FILE ##################################

# Read in configuration file
config_file = args.conf
config = read_json_config(config_file)

trainPercentage   = np.float64(config.get("training_data_percentage", 0.9))
lines_to_skip     = int(config.get("n_comment_lines", 1)) # number of lines to skip in the input file, not containing atom coordinates (empty lines do NOT count!)
epochs            = int(config.get("epochs_best_model", 2000)) # for best_model
callback_patience = int(config.get("callback_patience", 250)) # how many epoches without improvement are tolerated
dense_activ       = {'class_name': config.get("layer_activation_function", "leaky_softplus"), "config": {'alpha': np.float64(config.get("config_alpha_layer_activation_function", 0.03))}} # for MLP
final_activ       = config.get("final_activation_function", "linear") # activation function of the last layers of the MLPs 
loss_training     = config.get("training_loss_function", "mean_squared_error") # for trainig 

# Hyperparameter search inputs
hp_maxepochs = int(config.get("hp_epochs", 20)) # for tuner object 
hp_factor    = int(config.get("hp_factor", 2)) # for tuner object
hp_dict = {
    "neurons_min":    int(config.get("hp_neurons_min", 20)),
    "neurons_max":    int(config.get("hp_neurons_max", 100)),
    "neurons_step":   int(config.get("hp_neurons_step", 5)),
    "layers_min":     int(config.get("hp_layers_min", 2)),
    "layers_max":     int(config.get("hp_layers_max", 8)),
    "layers_step":    int(config.get("hp_layers_step", 1)),
    "regulizer":      config.get("hp_regulizer", "l2"),
    "learning_rates": config.get("hp_learning_rates", [1e-3, 5e-4, 1e-4, 5e-5]),
}

############################ PARAMETERS ##################################

# Normalization of geometries
norm = "const" #  const normalizes geometries once over all data ; 'batch' in batchs

# Other (BUGFIX)
clean_up_hpoutpath = True # delete hp_out_path before tuning (catch some errors 'oracle exited training' etc.)

# Paths and output
inputfile = args.file
outpath = os.getcwd() # Working directory
model_path = join(outpath, "best_model") # for model, params, unused indices 
hp_out_path = join(outpath, "outputtuner") # save tuner trials

# Callbacks for training the model
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

# Constants
EhtoeV, A2Bohr = unit_conversions["EhtoeV"], unit_conversions["A2Bohr"]

############################ START OF SCRIPT ##################################

## 1. Data Preparation

# Load and preprocess data
xyz_esp_data, energies, natoms, ntotal = load_data_excited_states_energies(inputfile, lines_to_skip) # energies and osc. str.
print(f"Number of Data points in the input file {inputfile} is {len(xyz_esp_data)}")

# Shuffle and split data
ntrain = int(ntotal * trainPercentage)
coords_train, esp_grads_train, targets_train, coords_test, esp_grads_test, targets_test = shuffle_and_split_train_test(xyz_esp_data, energies, ntrain)

# Extract ESP data
esp_train = esp_grads_train[:, :, 0] # esp_tmp can include esp + esp_grads
esp_test  = esp_grads_test[:, :, 0]

# Check whether ESP data is included in the training data
if esp_train.shape[1] == 0:
    esp_in_traindata = False
else:
    esp_in_traindata = True

# Store train and test data
data = {"coords": coords_train, "esp": esp_train, "targets": targets_train}

# Scaling
geomscaler = StandardScaler()
data["coords_mean"]   = geomscaler.fit(data["coords"].reshape(-1, 1)).mean_
data["coords_var"]    = geomscaler.var_
data["coords_scaled"] = (data["coords"] - data["coords_mean"]) / (data["coords_var"] ** 0.5)

targetscaler = StandardScaler()
data["targets_scaled"] = targetscaler.fit_transform(data["targets"].reshape(-1, 2))
data["targets_mean"]   = targetscaler.mean_
data["targets_var"]    = targetscaler.var_
target                 = data["targets_scaled"]

# Scale testdata with mean and var from traindata
coords_test_rescaled = (coords_test - data["coords_mean"]) / (data["coords_var"] ** 0.5)
    
# otput_spec with scaler being already fitted to energy -> Must come after scaler.fit
output_spec = { # this is ugly, as output_spec is needed in hp_simple_model()
    "QM/MM energy" : OutputSpec(from_subnets = "monolith",
                                scaler = targetscaler
                                )                                                
}

# Define x_train and y_train
if esp_in_traindata:
    x_train = [data["coords_scaled"], data["esp"]]
    y_train = data["targets_scaled"]
    callbacks = [stop_early, lr_reduction]
    y_test = [coords_test_rescaled, esp_test]
else:
    x_train = data["coords_scaled"]
    y_train = data["targets_scaled"]
    callbacks = [stop_early]
    y_test = coords_test_rescaled


## 2. Hyperparameter Search

if clean_up_hpoutpath: # removes directory which can be necessary
    if os.path.isdir(hp_out_path): # this is needed if tuner quits with "INFO:tensorflow:Oracle triggered exit"
        shutil.rmtree(hp_out_path) # = bash's rm -rf

# Initialize ModelBuilder
model_builder = hpModelBuilder_energy_oscStr(hp_dict, natoms, esp_in_traindata, dense_activ, final_activ, output_spec, loss_training, r2_metric, norm)

# Perform hyperparameter search
best_hps, tuner = model_builder.perform_hp_search(x_train, y_train, hp_maxepochs, hp_factor, callbacks, hp_out_path)

# print("------------------------------------------")
# print(f'''{best_hps.get("neurons")} neurons, {best_hps.get("layers")} layers, {best_hps.get("loss_ratio")} loss ratio, {best_hps.get("initial_lr")} initial learning rate and {best_hps.get("l2_penalty")} regulization penalty give the best results''')
# print("------------------------------------------")

# Build and train the best model
hp_model = tuner.hypermodel.build(best_hps)


## 3. Training best hp model

# pre-calculate geometries to fit the feat_std layers for normalizing inv.dists
if norm == "const": # eigentlich immer oder?
    feat_precomp = precompute_feature_in_chunks(data["coords_scaled"], hp_model, batch_size=32)  # changed for test
    # now the scaler of feat_std layer must be set. So weights and biases of this
    # layer must be so that x -> x-µ/std
    set_const_normalization_from_features(feat_precomp, hp_model)

# fit the models
hp_hist = hp_model.fit(x_train, target, epochs=epochs,
                     validation_split=0.1, verbose=2, callbacks=callbacks)

# Save model
hp_model.save(model_path)

# Get best epoch
hp_best_epoch = np.argmin(hp_hist.history["val_loss"])


## 4. Evaluation

# Evaluate the model
hp_pred = hp_model.predict(y_test)

# Scale the normalized prediction back to the natural scale of the data
hp_pred_scaled = targetscaler.inverse_transform(hp_pred)

# Scale the references for the test data to the normalized scale for evaluation
ref_scaled = targetscaler.transform(targets_test.reshape(-1,2))

# Get rescaled predictions in eV
energies_eV = targets_test[:, 0] * EhtoeV
hp_pred_scaled_eV = hp_pred_scaled[:, 0] * EhtoeV

# run keras model evaluation
hp_metrics = hp_model.evaluate(y_test, ref_scaled)

# Output important metrics    
print("\n\n", 22 * "-", "\n\t\tSummary\n", 22 * "-")
print("\n\tHP search lead to:\n", best_hps.get_config()["values"])

print("\n\tPerformance of HP model:")
test_mae_eV = mean_absolute_error(energies_eV, hp_pred_scaled_eV)
test_mae_osc = mean_absolute_error(targets_test[:, 1], hp_pred_scaled[:, 1])
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

# Save results
plt.hist(energies[:,1],bins=20)
plt.savefig(join(outpath, "osc_distribution_last.png"), dpi=300)

# Save energies to plot scatters
np.savetxt(join(model_path, 'hp_ref_energies_eV.dat'), energies_eV)
np.savetxt(join(model_path, 'hp_ref_osc.dat'), targets_train[:, 1])
np.savetxt(join(model_path, 'hp_predicted_energies_eV.dat'), hp_pred_scaled_eV)
np.savetxt(join(model_path, 'hp_predicted_osc.dat'), hp_pred_scaled[:,1])


## 5. Plotting
# Plot training progress
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
f.savefig(join(model_path, "test-loss.png"), dpi=300)
        
# Plot Scatter
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
fig.savefig(join(model_path, "scatter.png"), dpi=300)
plt.clf()
fig, ax = plt.subplots(1, figsize=(6,6))
ax.hist2d(targets_test[:, 1], hp_pred_scaled[:, 1],
                bins=1000, # Just for Tests; Mila wrote 1000
                cmin=1, # Just for Tests; Mila wrote 1 
                norm=mcolors.PowerNorm(0.5))
b, t = get_limits([targets_test[:, 1], hp_pred_scaled[:, 1]])
ax.set_xlabel("reference")
ax.set_ylabel("prediction")
ax.set_aspect("equal")
ax.set_ylim((b, t))
opti_ref = np.linspace(b, t, num=10)
ax.plot(opti_ref, opti_ref, c="C1")
fig.savefig(join(model_path, "scatter_osc.png"), dpi=300)

# save scaling parameters to GROMACS-readable format
hypers = [
    data["coords_mean"][0], 
    (data["coords_var"] ** 0.5)[0], 
    data["targets_mean"][0], 
    (data["targets_var"] ** 0.5)[0], 
    data["targets_mean"][1], 
    (data["targets_var"] ** 0.5)[1],
    (0,0) # no gradients
    
    ]

with open(os.path.join(model_path, "params.txt"), "w") as outf:
    for k in hypers:
        outf.write(f"{k}\n")
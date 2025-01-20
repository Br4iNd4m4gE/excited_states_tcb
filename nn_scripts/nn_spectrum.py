'''This script builds and trains a neural network to predict excitation/emission energies and oscillator strengths'''

import os
import sys
import numpy as np
import argparse
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
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
from pyNNsMD.esp_nn import OutputSpec, get_limits
from pyNNsMD.layers.wrapper import WrapEnergyModel

############################ PARSE ARGUMENTS ##################################

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
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

# Foldernames
best_hp_model_name = config.get("best_hp_model_name", "best_model")
hp_out_name        = config.get("hp_out_name", "outputtuner")

############################ PARAMETERS ##################################

# Normalization of geometries
norm = "const" #  "const" normalizes geometries once over all data; "batch" in batchs

# Other (BUGFIX)
clean_up_hpoutpath = True # delete hp_out_path before tuning (catch some errors 'oracle exited training' etc.)

# Paths and output
inputfile = args.file
outpath     = os.getcwd() # Working directory
model_path  = join(outpath, best_hp_model_name) # for model, params, unused indices 
hp_out_path = join(outpath, hp_out_name) # save tuner trials

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
EhtoeV = unit_conversions["EhtoeV"]

############################ START OF SCRIPT ##################################

## 1. Data Preparation

# Load and preprocess data
xyz_esp_data, energies, natoms, ntotal = load_data_excited_states_energies(inputfile, lines_to_skip) # energies and osc. str.
print(f"Number of Data points in the input file {inputfile} is {len(xyz_esp_data)}")

# Shuffle and split data
ntrain = int(ntotal * trainPercentage)
coords_train, esp_grads_train, targets_train, coords_test, esp_grads_test, targets_test = shuffle_and_split_train_test(xyz_esp_data, energies, ntrain)

# Extract ESP data
esp_train = esp_grads_train[:, :, 0] # esp_grads_train can include esp + esp_grads; esp_grads_train[:, :, 0] extracts only esp
esp_test  = esp_grads_test[:, :, 0]

# Check whether ESP data is included in the training data
if esp_train.shape[1] == 0:
    esp_in_traindata = False
else:
    esp_in_traindata = True

# Store train and test data
data = {"coords": coords_train, "esp": esp_train, "targets": targets_train}

# Scaling
targetscaler = StandardScaler()
data["targets_scaled"] = targetscaler.fit_transform(data["targets"].reshape(-1, 2))

# otput_spec with scaler being already fitted to energy -> Must come after scaler.fit
output_spec = { # this is ugly, as output_spec is needed in hp_simple_model()
    "QM/MM energy" : OutputSpec(from_subnets = "monolith",
                                scaler = targetscaler
                                )                                                
}

# Define x_train and y_train
if esp_in_traindata:
    x_train   = [data["coords"], data["esp"]]
    callbacks = [stop_early, lr_reduction]
    x_test    = [coords_test, esp_test]
else:
    x_train   = data["coords"]
    callbacks = [stop_early]
    x_test    = coords_test

y_train = data["targets_scaled"]

## 2. Hyperparameter Search

# Remove old outputtuner directory
if clean_up_hpoutpath:
    if os.path.isdir(hp_out_path): # this is needed if tuner quits with "INFO:tensorflow:Oracle triggered exit"
        shutil.rmtree(hp_out_path)

# Initialize ModelBuilder
model_builder = hpModelBuilder_energy_oscStr(hp_dict, natoms, esp_in_traindata, dense_activ, final_activ, output_spec, loss_training, r2_metric, norm, data["coords"])

# Perform hyperparameter search
best_hps, tuner = model_builder.perform_hp_search(x_train, y_train, hp_maxepochs, hp_factor, callbacks, hp_out_path)


## 3. Training best hp model

# Build the best model
hp_model = tuner.hypermodel.build(best_hps)

# Train the best model
hp_hist = hp_model.fit(x_train, data["targets_scaled"], epochs=epochs, validation_split=0.1, verbose=2, callbacks=callbacks)

# Wrap the model
wrapped_model = WrapEnergyModel(hp_model, targetscaler.mean_, targetscaler.var_)

# Single prediction (wrapped model returns the scaled back values in energy + oscillator strength)
model_pred = wrapped_model.predict(x_test) # you need to call the model once, before saving it

# Save model
wrapped_model.save(model_path)

# Get best epoch
hp_best_epoch_idx = np.argmin(hp_hist.history["val_loss"])


## 4. Evaluation of the model (stored in train.out)

# Get rescaled predictions in eV
model_pred_eV   = model_pred[:, 0]   * EhtoeV
energies_ref_eV = targets_test[:, 0] * EhtoeV

# Get oscillator strenghts
osc_pred = model_pred[:, 1]
osc_ref  = targets_test[:, 1]

# Calculate MAE
test_mae_eV  = mean_absolute_error(energies_ref_eV, model_pred_eV)
test_mae_osc = mean_absolute_error(osc_ref, osc_pred)

# Evaluate (scaled) model -> the model without the wrapper (true performance of model)
ref_scaled = targetscaler.transform(targets_test.reshape(-1,2))
hp_metrics = hp_model.evaluate(x_test, ref_scaled)
model_pred_scaled = hp_model.predict(x_test) # the direct output of the model (before the wrapper)

# Extract performance metrics
R2_total  = hp_metrics[2]
R2_energy = r2_score(ref_scaled[:, 0], model_pred_scaled[:, 0])
R2_osc    = r2_score(ref_scaled[:, 1], model_pred_scaled[:, 1])

# Get best hyperparameters
best_hps_config_str = "\t" + "\n\t".join([f"{key}: {value}" for key, value in best_hps.get_config()["values"].items()])

# Print performance metrics in train.out
print("\n\n", 70 * "-", "\n\t\t\t\t\t\t\t\tSummary\n", 70 * "-")
print("\n> Properties of final model")
print(best_hps_config_str)

print("\n> Performance of final model")
print(f"\ttest loss: {hp_metrics[0]} atomic units")
print(f"\ttest MAE energy: {test_mae_eV} eV")
print(f"\ttest MAE osc: {test_mae_osc}")
print(f"\ttest R2 energy: {R2_energy}\n")
print(f"\ttest R2 osc: {R2_osc}")
print(f"\ttest R2 (combined): {R2_total}")
print(f"\tbest epoch: {hp_best_epoch_idx}")

# Save data of energies and oscillator strengths of predictions and references
np.savetxt(join(model_path, 'model_ref_energies_eV.dat'), energies_ref_eV)
np.savetxt(join(model_path, 'model_predicted_energies_eV.dat'), model_pred_eV)
np.savetxt(join(model_path, 'model_ref_osc.dat'), osc_ref)
np.savetxt(join(model_path, 'model_predicted_osc.dat'), osc_pred)


## 5. Plotting

# Plot training progress
f, ax = plt.subplots(1, figsize=(6,6))
ax.plot(np.arange(len(hp_hist.history["loss"])), hp_hist.history["loss"], label="training loss")
ax.plot(np.arange(len(hp_hist.history["loss"])), hp_hist.history["val_loss"], label="validation loss")
ax.set_yscale("log")
ax.set_ylabel("log(MSE)")
ax.set_xlabel("epochs")
ax.plot(hp_best_epoch_idx, hp_hist.history["val_loss"][hp_best_epoch_idx], ls="", marker="x", ms=10, c="black", label="best model")
ax.axvline(x=hp_best_epoch_idx, color="gray", ls="--")
ax.axhline(y=hp_hist.history["val_loss"][hp_best_epoch_idx], color="gray", ls="--")
ax.legend()
f.savefig(join(model_path, "test_loss.png"), dpi=300)

# Plot Scatter Energies
fig, ax = plt.subplots(1, figsize=(6,6))
ax.hist2d(energies_ref_eV, model_pred_eV.flatten(), bins=1000, cmin=1, norm=mcolors.PowerNorm(0.5))
b, t = get_limits([energies_ref_eV, model_pred_eV])
ax.set_xlabel("reference [eV]")
ax.set_ylabel("prediction [eV]")
ax.set_aspect("equal")
ax.set_ylim((b, t))
opti_ref = np.linspace(b, t, num=10)
ax.plot(opti_ref, opti_ref, c="C1")
fig.savefig(join(model_path, "scatter.png"), dpi=300)

# Plot Scatter Oscillator Strengths
fig, ax = plt.subplots(1, figsize=(6,6))
ax.hist2d(osc_ref, osc_pred, bins=1000, cmin=1, norm=mcolors.PowerNorm(0.5))
b, t = get_limits([osc_ref, osc_pred])
ax.set_xlabel("reference")
ax.set_ylabel("prediction")
ax.set_aspect("equal")
ax.set_ylim((b, t))
opti_ref = np.linspace(b, t, num=10)
ax.plot(opti_ref, opti_ref, c="C1")
fig.savefig(join(model_path, "scatter_osc.png"), dpi=300)
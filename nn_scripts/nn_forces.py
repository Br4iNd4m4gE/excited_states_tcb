'''This script builds and trains a neural network for the prediction of intermolecular forces.
A hyperparameter search is done and the model is saved. The "mlmm_model" can be used
for the ML-MM GROMACS implementation.'''

import numpy as np
import sys
import argparse
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib as mpl
mpl.use('Agg')	#important for plotting while running on cluster

from os.path import join, isdir, isfile, dirname, abspath
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import unit_conversions, load_data_excited_states_forces, read_json_config
from pyNNsMD.nn_pes_src.device import set_gpu
from pyNNsMD.layers.wrapper import WrapForcesModel
from pyNNsMD.models.hp import hpModelBuilder_forces


############################ PARSE ARGUMENTS ##################################

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
# ap.add_argument("-p", "--outname")
ap.add_argument("-f", "--file", required=True, type=str, dest="file", action="store", help="Path to input file", metavar="file")
ap.add_argument("-c", "--conf", default=None, type=str, dest="conf", action="store", required=False, help="Path to config file, default: None", metavar="config")
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

############################ CONFIG FILE ##################################

# Read in configuration file
config_file = args.conf
config = read_json_config(config_file)
          
fit_epochs    = int(config.get("fit_epochs", 2000))
lines_to_skip = int(config.get("n_comment_lines", 1)) # number of comment lines to skip in the input file, not containing atom coordinates (empty lines do NOT count!)
batch_size    = int(config.get("batch_size", 128))

# Hyperparameter search inputs
hp_epochs = int(config.get("hp_epochs", 300))
hp_factor = int(config.get("hp_factor", 18))
hp_dict   = {
	"neurons_min":	int(config.get("hp_neurons_min", 200)),
	"neurons_max":	int(config.get("hp_neurons_max", 1000)),
	"neurons_step":	int(config.get("hp_neurons_step", 50)),
	"layers_min":	int(config.get("hp_layers_min", 2)),
	"layers_max":	int(config.get("hp_layers_max", 4)),
	"layers_step":	int(config.get("hp_layers_step", 1)),
	"initial_lr":	config.get("hp_initial_lr", [1e-3, 5e-4, 1e-4]),
	"l2_penalty":	config.get("hp_l2_penalty", [1e-3, 5e-4, 1e-4, 5e-5]),
	"loss_ratio":   config.get("hp_loss_ratio", [1e-2, 5e-3, 1e-3, 5e-4])
}

############################ PARAMETERS ##################################

# Inputs
inputfile = args.file

# Constants and Initializations
HaB_to_eVA, EhtoeV = unit_conversions["HaB_to_eVA"], unit_conversions["EhtoeV"]

stop_early = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 1e-7,
    patience = 100,
    verbose = 1,
    mode = 'auto',
    baseline = None,
    restore_best_weights = True
)

############################ START OF SCRIPT ##################################

## 1. Load data

# Load data
x, y, n_atoms, _ = load_data_excited_states_forces(inputfile, lines_to_skip)

# Generate train and test sets (y_train includes total energy and all forces)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, y_train, x_test, y_test = tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)

# Scale the output data
scaler = StandardScaler(with_std=False)	# without std the performance was better, distribution is already good apparently
scaler.fit(y_train)
scaler.mean_[1:] = 0 # no shift of forces
y_train_scaled, y_test_scaled = scaler.transform(y_train), scaler.transform(y_test)


## 2. Hyperparameter search

# Initialize ModelBuilder
model_builder = hpModelBuilder_forces(hp_dict, n_atoms, x_train)

# Perform hyperparameter search
best_hps, tuner = model_builder.perform_hp_search(x_train, y_train_scaled, hp_epochs, hp_factor, batch_size, stop_early)


## 3. Build and train the best model

# Build the best model
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
hist = best_model.fit(x_train, y_train_scaled, batch_size=batch_size, epochs=fit_epochs, verbose=2, validation_split=0.2)

# # Save the model
# best_model.save("best_model")

# Wrap the model for MLMM
mlmm_model = WrapForcesModel(best_model, scaler.mean_, 1.0)

# Single prediction (wrapped model returns the scaled back values in energy + oscillator strength)
pred_rescaled = mlmm_model(x_test)

# Save the wrapped model
mlmm_model.save("mlmm_model")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")


## 4. Evaluation of the model (stored in train.out)

# Energy predictions
energy_test = y_test[:,0]
energy_pred = pred_rescaled[:,0]

# Convert energies to eV
energy_test_ev = energy_test * EhtoeV
energy_pred_ev = energy_pred * EhtoeV

# Flatten the force predictions
forces_pred = K.flatten(pred_rescaled[:, 1:])
forces_test = K.flatten(y_test[:, 1:])

# Convert forces to eV/A
forces_test_evA = forces_test * HaB_to_eVA
forces_pred_evA = forces_pred * HaB_to_eVA

# Calculate R2 and MAE
mae_te = mean_absolute_error(y_test[:, 0], pred_rescaled[:, 0])
mae_forces = mean_absolute_error(forces_test, forces_pred)

# Convert MAE to eV/A
mae_te_eV = mae_te * EhtoeV
mae_forces_eV = mae_forces * HaB_to_eVA

# Calculate R2 score
r2_te = r2_score(y_test[:, 0], pred_rescaled[:, 0])
r2_forces = r2_score(forces_test, forces_pred)

# Get the standard deviation of the forces
force_std = np.mean(np.std(y_train[:, 1:]))

# Get the loss curves
losses = hist.history["loss"]
val_losses = hist.history["val_loss"]

# Print results in train.out
print("\n\n", 70 * "-", "\n\t\t\t\t\t\t\t\tSummary\n", 70 * "-")
print(f'\n> Network Architecture:\n{best_hps.get("neurons")} neurons, {best_hps.get("layers")} layers, {best_hps.get("loss_ratio")} loss ratio, {best_hps.get("initial_lr")} initial learning rate and {best_hps.get("l2_penalty")} regulization penalty give the best results\n\n')

print(f"> R2 Total Energy:\n\t{r2_te}")
print(f"> MAE Total Energy:\n\t{mae_te} Eh\n\t{mae_te_eV} eV")
print("-----")
print(f"> R2 Forces:\n\t{r2_forces}")
print(f"> MAE Forces:\n\t{mae_forces} Eh/B\n\t{mae_forces_eV} eV/A")
print("-----")
print(f"> MAE Forces / STD Forces:\n\t{100 * mae_forces / force_std} %\n")

# Save predictions and references for test data
np.savetxt("energy_predictions_Eh.txt", energy_pred)
np.savetxt("force_predictions_EhB.txt", forces_pred)
np.savetxt("energy_ref.txt_Eh", energy_test)
np.savetxt("force_ref_EhB.txt", forces_test)


## 5. Plotting

# Loss curve
ep = np.arange(1, len(losses) + 1)
plt.semilogy(ep, losses, label="loss")
plt.semilogy(ep, val_losses, label="val_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Combined MSE")
plt.savefig("loss.png", dpi=300)
plt.clf()

# Plot total energy
plt.hist2d(energy_test, energy_pred, bins=100, cmin=1, cmap='inferno')
plt.xlabel('True Values [Eh]')
plt.ylabel('Predictions [Eh]')
plt.colorbar()
plt.plot([min(y_test[:,0]), max(y_test[:,0])], [min(y_test[:,0]), max(y_test[:,0])])
plt.savefig("tot_ene.png", dpi=300)
plt.clf()

# Plot forces histogram with minimum in bin of 1
plt.hist2d(forces_test_evA, forces_pred_evA, bins=100, cmin=1, cmap='inferno')
plt.xlabel('True Values [eV/A]')
plt.ylabel('Predictions [eV/A]')
plt.colorbar()
plt.plot([-13.5,13.5], [-13.5,13.5])
plt.savefig("forces_cmin1.png", dpi=300)
plt.clf()

# Plot forces histogram with minimum in bin of 50
plt.hist2d(forces_test_evA, forces_pred_evA, bins=100, cmin=50, cmap='inferno')
plt.xlabel('True Values [eV/A]')
plt.ylabel('Predictions [eV/A]')
plt.colorbar()
plt.plot([-13.5,13.5], [-13.5,13.5])
plt.savefig("forces_cmin50.png", dpi=300)
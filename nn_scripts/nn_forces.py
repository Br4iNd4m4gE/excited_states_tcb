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
from pyNNsMD.layers.normalize import NormalizationLayer
from pyNNsMD.layers.inverse_distance import InverseDistance, FirstInverseDistance
from pyNNsMD.models.hp import hpModelBuilder


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
AtoBohr, HaB_to_eVA = unit_conversions["A2Bohr"], unit_conversions["HaB_to_eVA"]

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

# Load data
x, y, n_atoms, _ = load_data_excited_states_forces(inputfile, lines_to_skip)

# Generate train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, y_train, x_test, y_test = tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)

# Get mask to filter large distances, scale output data and get input mean and variance for Normalization
first_preprocessor   = FirstInverseDistance()
dummy, full_mask     = first_preprocessor(x_train)	#fullmask has True or False values for all distance checks in all samples
reduced_mask         = tf.math.reduce_all(full_mask,0)	#if the distance is below a cutoff for all samples that distance is always considered
#reducedmask          = tf.math.reduce_any(fullmask,0)	#if the distance is below a cutoff for any sample that distance is always considered
preprocessor         = InverseDistance(reduced_mask)	#initialize inverse distance and filtering layer
xtrain_dist          = preprocessor(x_train)
dist_shape           = np.shape(xtrain_dist)
x_train_dist_mean    = np.mean(xtrain_dist, 0)
dist_mean            = np.mean(x_train_dist_mean[:-n_atoms])	#different means for inverse distances and ESP
esp_mean             = np.mean(x_train_dist_mean[-n_atoms:])
norm_mean            = np.ones(dist_shape[1])
norm_mean[:-n_atoms] = dist_mean * norm_mean[:-n_atoms]
norm_mean[-n_atoms:] = esp_mean * norm_mean[-n_atoms:]
x_train_dist_var     = np.var(xtrain_dist, 0)
dist_var             = np.mean(x_train_dist_var[:-n_atoms])	#different means for inverse distances and ESP
esp_var              = np.mean(x_train_dist_var[-n_atoms:])
norm_var             = np.ones(dist_shape[1])
norm_var[:-n_atoms]  = dist_var * norm_var[:-n_atoms]
norm_var[-n_atoms:]  = esp_var * norm_var[-n_atoms:]
normalizer           = NormalizationLayer(norm_mean, norm_var)	#initialize normalization layer
scaler               = StandardScaler(with_std=False)	#without std the performance was better, distribution is already good apparently
scaler.fit(y_train) #y_train includes total energy and all forces
force_std            = np.mean(np.std(y_train[:,1:]))
#scaler.var_[1:]      = scaler.var_[0] #F=dE_tot/dx=dE_elec/dx+dE_rep/dx must hold so the scaling scaled=(raw-mean)/sqrt(var) must be coherent
scaler.mean_[1:]     = 0	#gradients should not be changed shifted
y_train_scaled       = scaler.transform(y_train)
y_test_scaled        = scaler.transform(y_test)

# Initialize ModelBuilder
model_builder = hpModelBuilder(hp_dict, n_atoms, preprocessor, normalizer)

# Perform hyperparameter search
best_hps, tuner = model_builder.perform_hp_search(x_train, y_train_scaled, hp_epochs, hp_factor, batch_size, stop_early)

print("------------------------------------------")
print(f'''{best_hps.get("neurons")} neurons, {best_hps.get("layers")} layers, {best_hps.get("loss_ratio")} loss ratio, {best_hps.get("initial_lr")} initial learning rate and {best_hps.get("l2_penalty")} regulization penalty give the best results''')
print("------------------------------------------")

# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
hist = best_model.fit(x_train, y_train_scaled, batch_size=batch_size, epochs=fit_epochs, verbose=2, validation_split=0.2)
losses = hist.history["loss"]
val_losses = hist.history["val_loss"]
# eval = best_model.evaluate(x_test, y_test_scaled)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(best_model(x_test).shape, x_test.shape, y_test_scaled.shape)

#Save models
best_model.save("best_model")
mlmm_model = WrapForcesModel(best_model, scaler.mean_, 1.0)
test_pred_rescaled = mlmm_model(x_test)
mlmm_model.save("mlmm_model")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

#Print Performance
forces_pred = K.flatten(test_pred_rescaled[:, 1:])
forces_test = K.flatten(y_test[:, 1:])
print("R2 Total Energy:")
print(r2_score(y_test[:, 0], test_pred_rescaled[:, 0]))
print("MAE Total Energy:")
mae_te = mean_absolute_error(y_test[:, 0], test_pred_rescaled[:, 0])
print(mae_te, ' eV')
print("R2 Forces:")
print(r2_score(forces_test, forces_pred))
print("MAE Forces:")
mae_forces = mean_absolute_error(forces_test, forces_pred)
print(mae_forces, ' eV/A')
print("MAE Forces/STD Forces in %:")
print(100 * mae_forces / force_std)

#Save predictions and references for test data
np.savetxt("energy_predictions.txt", test_pred_rescaled[:, 0])
np.savetxt("force_predictions.txt", forces_pred)
np.savetxt("energy_ref.txt", y_test[:, 0])
np.savetxt("force_ref.txt", forces_test)

#Plot
#Loss curve
ep = np.arange(1, len(losses) + 1)
plt.semilogy(ep,losses, label="loss")
plt.semilogy(ep, val_losses, label="val_loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Combined MSE")
plt.savefig("loss.png", dpi=300)
plt.clf()

#Total Energy
plt.hist2d(y_test[:,0], test_pred_rescaled[:,0], bins=100, cmin=1, cmap='inferno')
plt.xlabel('True Values [Eh]')
plt.ylabel('Predictions [Eh]')
plt.colorbar()
plt.plot([min(y_test[:,0]), max(y_test[:,0])], [min(y_test[:,0]), max(y_test[:,0])])
plt.savefig("tot_ene.png", dpi=300)
plt.clf()

#Forces 1
plt.hist2d(forces_test * HaB_to_eVA, forces_pred * HaB_to_eVA, bins=100, cmin=1, cmap='inferno')
plt.xlabel('True Values [eV/A]')
plt.ylabel('Predictions [eV/A]')
plt.colorbar()
plt.plot([-13.5,13.5], [-13.5,13.5])
plt.savefig("forces_cmin1.png", dpi=300)
plt.clf()

#Forces 50
plt.hist2d(forces_test * HaB_to_eVA, forces_pred * HaB_to_eVA, bins=100, cmin=50, cmap='inferno')
plt.xlabel('True Values [eV/A]')
plt.ylabel('Predictions [eV/A]')
plt.colorbar()
plt.plot([-13.5,13.5], [-13.5,13.5])
plt.savefig("forces_cmin50.png", dpi=300)
plt.clf()

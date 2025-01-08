'''This script builds and trains a neural network to predict excitation/emission energies and oscillator strengths'''

import os
import sys
import numpy as np
import argparse
from os.path import join, isdir, isfile, dirname, abspath
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

from itertools import combinations
import tensorflow.keras as ks
import kerastuner as kt
from sklearn.preprocessing import StandardScaler
# sys.path.append("/home/cschmidt/bin")
sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import parse_single_file, shuffle_and_split, generate_invd_list, get_file_length, extract_number_of_atoms
from pyNNsMD.nn_pes_src.device import set_gpu
from pyNNsMD.utils.loss import r2_metric
from pyNNsMD.layers.mlp import MLP
from pyNNsMD.layers.features import FeatureGeometric
from pyNNsMD.layers.normalize import ConstLayerNormalization

from esp_nn import precompute_feature_in_chunks, set_const_normalization_from_features
from esp_nn import OutputSpec, SubNet, build_model, get_limits
from esp_nn import build_geom_preprocess_layer, ScaledMeanAbsoluteError
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
# traindata = "nn_both_isomers_cleaned_delta.txt" # full path of traindata
traindata = args.file
# outpath="/data/cschmidt/excited_states/data_energy" # Here stuff is written
outpath = os.getcwd() # Here stuff is written
mod_outpath=join(outpath, "best_model") # for model, params, unused indices 
hp_outpath=join(outpath, "outputtuner") # save tuner trials
delete_tunertrials = False # delete the hp_outpath after script finished
logfile = join(outpath, "logfile.txt") # writes all stdout into this file

# training data and scenario
# natoms = 54
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
neurons_min = 20 # only hp-model
neurons_max = 100 # only hp-model
# neurons_min = neurons_max = 30 for best model so far
neurons_step = 5 # only hp-model
layers_min = 2 # only hp-model
layers_max = 8 # only hp-model
# layesr_min = layers_max = 2 for best model so far
layers_step = 1 # only hp-model
regulizer = "l2" # only hp-model
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5] # only hp-model
#learning_rates = [1e-4] for best model so far
hp_maxepochs = 20 # for tuner object 
hp_factor = 2 # for tuner object

# original-esp model to compare to
create_origmodel = True # one model without HP search is trained and compared
origmod_outpath=join(outpath, "orig_best_model") # for model, params, unused indices 
orig_neurons = 30 # neurons in mlp layer in orig-model
orig_learning_rate = 1e-4 # learning rate of orig-model
orig_layer_depth = 2 # depth of mlp layer for oig-model

# plotting
plot_scatters = True # plot predicted_energy(eV) vs. ref_energy (eV) 
plot_learning_curve = True # plot log(mse) vs. epochs

# other
clean_up_hpoutpath = True # delet hp_outpath before tuning (catch some errors 'oracle exited training' etc.)
########################  End of User section  ################################

def hp_simple_model_site(hp):
    # hp = kt.HyperParameters() must be given if single model shall be constructed
    """ Building up the esp-model for hyperparametersearch. This is a modified 
    verion of esp_nn.py/build_model() with hyperparameters. Therefore, the
    main architecture is fixed. You may vary number of neurons in MLP layer,
    the depth, the l1 or l2 regularization and the learning rate for now."""
    # 0. create interatomic distance matrix
    geom_idx = [(i,j) for i,j in combinations(range(natoms), 2)] #len=3570
    interatomic_dists = np.array(geom_idx) # this is [ [0,1],[0,2],...,[83,84] ]
    # 1. Geom in and Geom Prep (later already has feat_std)
    geom_in, geom_prep = build_geom_preprocess_layer(natoms, 
                                                     interatomic_dists, 
                                                     norm=norm)
    # 2. Esp_in 
    esp_in = ks.Input(shape=(natoms,), dtype='float32', name='esp_input')
    # 3. Concat
    rep = ks.layers.Concatenate(name="concat_layer")([geom_prep, esp_in])
    # 4. MLP with HP search
    neurons = hp.Int("nn_size", neurons_min, neurons_max, neurons_step)
    hp_layer_depth = hp.Int("depth", layers_min, layers_max, layers_step)
    hp_regularizer = regulizer
    mlp = MLP(dense_units=neurons, 
              dense_depth=hp_layer_depth, 
              dense_activ=dense_activ, 
              dense_activ_last=dense_activ,
              dense_kernel_regularizer=hp_regularizer,
              name="monolith")
    # 5. Output
    res = mlp(rep)
    final_layer = ks.layers.Dense(2, 
                    activation=final_activ, 
                    use_bias=True, 
                    name="out_vom_mlp")(res)
    # summary 
    inputs_list = [geom_in, esp_in]
    outputs_list = [final_layer]
    # make a model out of it all
    model = ks.Model(inputs=inputs_list, outputs=outputs_list)
    hp_learning_rate = hp.Choice("learning_rate", values=learning_rates)
    opti = ks.optimizers.Adam(learning_rate=hp_learning_rate)
    # get metrics offenes ToDo !
    for name, o in output_spec.items():
        # if targets are scaled, the MAE must be converted to original data units
        maes=[]
        if o.scaler:
            mae_scaled = ScaledMeanAbsoluteError(scaling_shape=o.scaler.scale_.shape)
            mae_scaled.set_scale(o.scaler.scale_)
            maes.append(mae_scaled)
        else:
            maes.append("mean_absolute_error")
    # final model configuration
    model.compile(optimizer=opti, 
                  loss = loss,
                  metrics=[[mae, r2_metric] for mae in maes]) # for history and tuning
    return model

def hp_simple_model_site_noesp(hp):
    # hp = kt.HyperParameters() must be given if single model shall be constructed
    """ Building up the esp-model for hyperparametersearch. This is a modified 
    verion of esp_nn.py/build_model() with hyperparameters. Therefore, the
    main architecture is fixed. You may vary number of neurons in MLP layer,
    the depth, the l1 or l2 regularization and the learning rate for now."""
    # 0. create interatomic distance matrix
    geom_idx = [(i,j) for i,j in combinations(range(natoms), 2)] #len=3570
    interatomic_dists = np.array(geom_idx) # this is [ [0,1],[0,2],...,[83,84] ]
    # 1. Geom in and Geom Prep (later already has feat_std)
    geom_shape = (natoms, 3)
    geom_in = ks.Input(shape=geom_shape, dtype='float32', name='geo_input')
    feat_layer = FeatureGeometric(invd_shape = interatomic_dists.shape, 
                                  name="feat_layer")
    # which interatomic distances to use
    feat_layer.set_mol_index(interatomic_dists, None, None)
    full = feat_layer(geom_in)
    full = ConstLayerNormalization(name="feat_std")(full)
    # 2. Esp_in 
    # esp_in = ks.Input(shape=(natoms,), dtype='float32', name='esp_input')
    # # 3. Concat
    # rep = ks.layers.Concatenate(name="concat_layer")([geom_prep, esp_in])
    # 4. MLP with HP search
    neurons = hp.Int("nn_size", neurons_min, neurons_max, neurons_step)
    hp_layer_depth = hp.Int("depth", layers_min, layers_max, layers_step)
    hp_regularizer = hp.Choice("use_reg_weight", values=regulizers)
    mlp = MLP(dense_units=neurons, 
              dense_depth=hp_layer_depth, 
              dense_activ=dense_activ, 
              dense_activ_last=dense_activ,
              dense_kernel_regularizer=hp_regularizer,
              name="monolith")
    # 5. Output
    res = mlp(full)
    final_layer = ks.layers.Dense(1, 
                    activation=final_activ, 
                    use_bias=True, 
                    name="out_vom_mlp")(res)
    # summary 
    inputs_list = [geom_in]
    outputs_list = [final_layer]
    # make a model out of it all
    model = ks.Model(inputs=inputs_list, outputs=outputs_list)
    hp_learning_rate = hp.Choice("learning_rate", values=learning_rates)
    opti = ks.optimizers.Adam(learning_rate=hp_learning_rate)
    # get metrics offenes ToDo !
    for name, o in output_spec.items():
        # if targets are scaled, the MAE must be converted to original data units
        maes=[]
        if o.scaler:
            mae_scaled = ScaledMeanAbsoluteError(scaling_shape=o.scaler.scale_.shape)
            mae_scaled.set_scale(o.scaler.scale_)
            maes.append(mae_scaled)
        else:
            maes.append("mean_absolute_error")
    # final model configuration
    model.compile(optimizer=opti, 
                  loss = loss,
                  metrics=[[mae, r2_metric] for mae in maes]) # for history and tuning
    return model

def hp_simple_model_cpl(hp):
    # hp = kt.HyperParameters() must be given if single model shall be constructed
    """ Building up the esp-model for hyperparametersearch. This is a modified 
    verion of esp_nn.py/build_model() with hyperparameters. Therefore, the
    main architecture is fixed. You may vary number of neurons in MLP layer
    and the learning rate for now."""
    # 0. create interatomic distance matrix
    interatomic_dists = generate_invd_list("cpl", "inter", natoms)
    interatomic_dists = np.array(interatomic_dists) #shape (7225,2)
    # 1. Geom in and Geom Prep (later already has feat_std)
    geom_shape = (2*natoms, 3)
    geom_in = ks.Input(shape=geom_shape, dtype='float32', name='geo_input')
    feat_layer = FeatureGeometric(invd_shape = interatomic_dists.shape, 
                                  name="feat_layer")
    # which interatomic distances to use
    feat_layer.set_mol_index(interatomic_dists, None, None)
    full = feat_layer(geom_in)
    full = ConstLayerNormalization(name="feat_std")(full)
    # 4. MLP with HP search
    neurons = hp.Int("nn_size", neurons_min, neurons_max, neurons_step)
    hp_layer_depth = hp.Int("depth", layers_min, layers_max, layers_step)
    hp_regularizer = hp.Choice("use_reg_weight", values=regulizers)
    mlp = MLP(dense_units=neurons, 
              dense_depth=hp_layer_depth, 
              dense_activ=dense_activ, 
              dense_activ_last=dense_activ,
              dense_kernel_regularizer=hp_regularizer,
              name="monolith")
    # 5. Output
    res = mlp(full)
    final_layer = ks.layers.Dense(1, 
                    activation=final_activ, 
                    use_bias=True, 
                    name="out_vom_mlp")(res)
    # summary 
    inputs_list = [geom_in]
    outputs_list = [final_layer]
    # make a model out of it all
    model = ks.Model(inputs=inputs_list, outputs=outputs_list)
    hp_learning_rate = hp.Choice("learning_rate", values=learning_rates)
    opti = ks.optimizers.Adam(learning_rate=hp_learning_rate)
    # get metrics offenes ToDo !
    for name, o in output_spec.items():
        # if targets are scaled, the MAE must be converted to original data units
        maes=[]
        if o.scaler:
            mae_scaled = ScaledMeanAbsoluteError(scaling_shape=o.scaler.scale_.shape)
            mae_scaled.set_scale(o.scaler.scale_)
            maes.append(mae_scaled)
        else:
            maes.append("mean_absolute_error")
    # final model configuration
    model.compile(optimizer=opti, 
                  loss = loss,
                  metrics=[[mae, r2_metric] for mae in maes]) # for history and tuning
    return model

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
# david logger end

###########################  Start of Sript  ##################################

##### 0. Constants and Definitions
Ha2eV = 27.211396132
A2Bohr = 1.8897259886
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
    coords *=A2Bohr
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
data["x_mean"] = geoscaler.fit(data["x"].reshape(-1,1)).mean_
data["x_var"] = geoscaler.var_
data["x_scaled"]= (data["x"]-data["x_mean"])/(data["x_var"]**0.5) 
# same with energy
targetscaler = StandardScaler()
data["targets_scaled"] = targetscaler.fit_transform(data["targets"].reshape(-1,2))
data["targets_mean"] = targetscaler.mean_
data["targets_var"] = targetscaler.var_
target = data["targets_scaled"]
    
# otput_spec with scaler being already fitted to energy -> Must come after scaler.fit
output_spec = { # this is ugly, as output_spec is needed in hp_simple_model()
    "QM/MM energy" : OutputSpec(from_subnets = "monolith",
                                scaler = targetscaler
                                )                                                
} 

##### 2. Hyperparameter Search
# this is needed if tuner quits with "INFO:tensorflow:Oracle triggered exit"
if clean_up_hpoutpath: # removes directory which can be necessary 
    import shutil 
    if os.path.isdir(hp_outpath):
        shutil.rmtree(hp_outpath) # = bash's rm -rf
    
# Instantiate tuner
if esp_in_traindata:
    tuner = kt.Hyperband(
        hp_simple_model_site,
        #hypermodel=hp_simple_model,
        # objective = kt.Objective("val_r2_metric", "max"), # müsste gehen! 
        # objective = "loss", # geht! Aber nicht val_loss
        objective=kt.Objective("val_r2_metric", "max"),
        #max_trials = 3,
        max_epochs = hp_maxepochs,
        factor = hp_factor,
        directory = hp_outpath)
elif not esp_in_traindata:
    tuner = kt.Hyperband(
        hp_simple_model_site_noesp,
        #hypermodel=hp_simple_model,
        # objective = kt.Objective("val_r2_metric", "max"), # müsste gehen! 
        # objective = "loss", # geht! Aber nicht val_loss
        objective=kt.Objective("val_r2_metric", "max"),
        #max_trials = 3,
        max_epochs = hp_maxepochs,
        factor = hp_factor,
        directory = hp_outpath)

# control
print("\n\tSearch Space Summary:")
print(tuner.search_space_summary())

# actual search
if esp_in_traindata:
    tuner.search(x = [data["x_scaled"], data["esp"]], 
                y = data["targets_scaled"], 
                verbose =2,
                epochs = hp_maxepochs,
                validation_split=0.1, 
                callbacks= [stop_early,lr_reduction],
                batch_size= 64 
                )
elif not esp_in_traindata:
    tuner.search(x = data["x_scaled"], 
                y = data["targets_scaled"], 
                verbose =1,
                epochs = hp_maxepochs,
                validation_split=0.1, 
                callbacks= [stop_early],
                batch_size= 64 
                )

##### 3. Building Models
# set best hyperparams to best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
hp_model = tuner.hypermodel.build(best_hps)

# for comparison build the default classical nn-esp model. Some input is needed:
if create_origmodel:
    # model needs interatomic dists
    if tgt == "site":
        geom_idx = [(i,j) for i,j in combinations(range(natoms), 2)]
        interatomic_dists = np.array(geom_idx) # this is [ [0,1],[0,2],...,[83,84] ]
    elif tgt == "cpl":
        interatomic_dists = generate_invd_list("cpl", "inter", natoms)
        interatomic_dists = np.array(interatomic_dists) #shape (7225,2)
        
    # model needs subnets
    if esp_in_traindata:
        subnets = {
            "monolith" : SubNet(neurons = orig_neurons, # number of neurons per hidden dense layer
                                depth = orig_layer_depth, # number of fully connected hidden layers
                                activ=dense_activ, # activation function for hidden layers
                                final_activ = final_activ, # act func for final subnet output layer
                                num_outputs = 2, # number of output neurons of MLP subunit
                                rep = "both") # 'geom' assumes (natoms,3) input, erg preprocessing
            }
    else:
        subnets = {
            "monolith" : SubNet(neurons = orig_neurons, # number of neurons per hidden dense layer
                                depth = orig_layer_depth, # number of fully connected hidden layers
                                activ=dense_activ, # activation function for hidden layers
                                final_activ = final_activ, # act func for final subnet output layer
                                num_outputs = 1, # number of output neurons of MLP subunit
                                rep = "geom") # 'geom' assumes (natoms,3) input, erg preprocessing
            }    
        
    # build orig_model
    orig_model = build_model(subnets, output_spec, natoms, interatomic_dists,
                        norm=norm, loss=loss,
                        learning_rate = orig_learning_rate, print_summary=False,
                        make_subnet_predictions_accessible=False)


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
    if create_origmodel:
        set_const_normalization_from_features(feat_precomp, orig_model)

# fit the models
if esp_in_traindata:
    hp_hist = hp_model.fit([data["x_scaled"], data["esp"]], target, epochs=epochs,
                     validation_split=0.1, verbose=2, callbacks=[stop_early,lr_reduction])
    if create_origmodel:
        orig_hist = orig_model.fit([data["x_scaled"], data["esp"]], target, epochs=epochs,
                            validation_split=0.1, verbose=2, callbacks=[stop_early,lr_reduction])
elif not esp_in_traindata:
    hp_hist = hp_model.fit(data["x_scaled"], target, epochs=epochs,
                     validation_split=0.1, verbose=2, callbacks=[stop_early])
    if create_origmodel:
        orig_hist = orig_model.fit(data["x_scaled"], target, epochs=epochs,
                            validation_split=0.1, verbose=2, callbacks=[stop_early])

# Save model
hp_model.save(mod_outpath)
if create_origmodel: orig_model.save(origmod_outpath)

# Get best epoch
hp_best_epoch = np.argmin(hp_hist.history["val_loss"])
if create_origmodel: orig_best_epoch = np.argmin(orig_hist.history["val_loss"])

##### 5. Prediction
# Scale testdata with mean and var from traindata
testcoords=(coords[unused[:ntest]]-data["x_mean"])/(data["x_var"]**0.5)
# Predictions
if  esp_in_traindata:
    hp_pred = hp_model.predict([testcoords,esp_raw[unused[:ntest]]])
    if create_origmodel: orig_pred = orig_model.predict([testcoords,esp_raw[unused[:ntest]]])
elif not esp_in_traindata:
    hp_pred = hp_model.predict(testcoords)
    if create_origmodel: orig_pred = orig_model.predict(testcoords)
    
##### 6. Evaluation
# scale the normalized prediction back to the natural scale of the data
hp_pred_scaled = targetscaler.inverse_transform(hp_pred)
if create_origmodel: orig_pred_scaled = targetscaler.inverse_transform(orig_pred)
# scale the references for the test data to the normalized scale for evaluation
ref_scaled = targetscaler.transform(energies[unused[:ntest]].reshape(-1,2))

# get rescaled predictions in eV
energies_eV=energies[unused[:ntest],0]*27.2114
hp_pred_scaled_eV=hp_pred_scaled[:,0]*27.2114
if create_origmodel: orig_pred_scaled_eV=orig_pred_scaled[:,0]*27.2114

# run keras model evaluation
if esp_in_traindata:
    hp_metrics = hp_model.evaluate([testcoords, esp_raw[unused[:ntest]]], ref_scaled)
    if create_origmodel: orig_metrics = orig_model.evaluate([testcoords,esp_raw[unused[:ntest]]], ref_scaled)
elif not esp_in_traindata:
    hp_metrics = hp_model.evaluate(testcoords, ref_scaled)
    if create_origmodel: orig_metrics = orig_model.evaluate(testcoords, ref_scaled)

# Output important metrics    
print("\n\n", 22 * "-", "\n\t\tSummary\n", 22 * "-")
print("\n\tHP search lead to:\n", best_hps.get_config()["values"])

print("\n\tPerformance of HP model:")
test_mae_eV = mean_absolute_error(energies_eV,hp_pred_scaled_eV)
test_mae_osc = mean_absolute_error(energies[unused[:ntest],1],hp_pred_scaled[:,1])
print("test loss: ", hp_metrics[0])
# print("test MAE (atomic): ", hp_metrics[1])
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

if create_origmodel:
    print("\n\tPerformance of original model:")
    print("This is a model trained without a HP search and its only use is for comparison.")
    orig_test_mae_eV = mean_absolute_error(energies_eV,orig_pred_scaled_eV)
    orig_test_mae_osc = mean_absolute_error(energies[unused[:ntest],1],orig_pred_scaled[:,1])
    print("test loss: ", orig_metrics[0])
    # print("test MAE (atomic): ", orig_metrics[1])
    print("test MAE (eV): ", orig_test_mae_eV)
    print("test MAE Osc: ", orig_test_mae_osc)
    print("test R2: ", orig_metrics[2])
    print(f"best epoch: {orig_best_epoch}")
    print("---")
    print("best train loss (atomic): ", orig_hist.history["loss"][orig_best_epoch])
    print("best train R2: ", orig_hist.history["r2_metric"][orig_best_epoch])
    print("---")
    print("best val loss (atomic): ", orig_hist.history["val_loss"][orig_best_epoch])
    print("best val R2: ", orig_hist.history["val_r2_metric"][orig_best_epoch])

##### 7. Write information to files
# save indices not used for training to file for later use in tests
np.savetxt(join(mod_outpath, "indices_for_testing_all.txt"), unused)
np.savetxt(join(mod_outpath, "indices_for_testing_testdata.txt"), unused[:ntest])
if create_origmodel:
    np.savetxt(join(origmod_outpath, "indices_for_testing_all.txt"), unused)
    np.savetxt(join(origmod_outpath, "indices_for_testing_testdata.txt"), unused[:ntest])
#check distribution
plt.hist(energies[:,1],bins=20)
plt.savefig(join(outpath, "osc_distribution_last.png"), dpi=300)

# save energies to plot scatters
np.savetxt(join(mod_outpath, 'hp_ref_energies_eV.dat'), energies_eV)
np.savetxt(join(mod_outpath, 'hp_ref_osc.dat'), energies[unused[:ntest],1])
np.savetxt(join(mod_outpath, 'hp_predicted_energies_eV.dat'), hp_pred_scaled_eV)
np.savetxt(join(mod_outpath, 'hp_predicted_osc.dat'), hp_pred_scaled[:,1])
if create_origmodel:
    np.savetxt(join(origmod_outpath, 'hp_ref_energies_eV.dat'), energies_eV)
    np.savetxt(join(origmod_outpath, 'hp_ref_osc.dat'), energies[unused[:ntest],1])
    np.savetxt(join(origmod_outpath, 'hp_predicted_energies_eV.dat'), orig_pred_scaled_eV)
    np.savetxt(join(origmod_outpath, 'hp_predicted_osc.dat'), orig_pred_scaled[:,1])
    
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
    #plt.show()
    
    # same for orig_model
    if create_origmodel:
        f, ax = plt.subplots(1, figsize=(6,6))
        ax.plot(np.arange(len(orig_hist.history["loss"])), orig_hist.history["loss"], label="training loss")
        ax.plot(np.arange(len(orig_hist.history["loss"])), orig_hist.history["val_loss"], label="validation loss")
        ax.set_yscale("log")
        ax.set_ylabel("log(MSE)")
        ax.set_xlabel("epochs")
        ax.plot(orig_best_epoch, orig_hist.history["val_loss"][orig_best_epoch], ls="", marker="x", ms=10, c="black", label="best model")
        ax.axvline(x=orig_best_epoch, color="gray", ls="--")
        ax.axhline(y=orig_hist.history["val_loss"][orig_best_epoch], color="gray", ls="--")
        ax.legend()
        f.savefig(join(origmod_outpath, "test-loss.png"), dpi=300)
        #plt.show()
        
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

        
    # same for orig_model
    if create_origmodel:
        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.hist2d(energies_eV, orig_pred_scaled_eV.flatten(), 
                  bins=1000, # Just for Tests; Mila wrote 1000
                  cmin=1,  # Just for Tests; Mila wrote 1 
                  norm=mcolors.PowerNorm(0.5))
        b,t = get_limits([energies_eV, orig_pred_scaled_eV])
        ax.set_xlabel("reference (eV)")
        ax.set_ylabel("prediction (eV)")
        ax.set_aspect("equal")
        ax.set_ylim((b,t))
        opti_ref = np.linspace(b,t,num=10)
        ax.plot(opti_ref, opti_ref, c="C1")
        fig.savefig(join(origmod_outpath, "scatter.png"), dpi=300)
        plt.clf()
        fig, ax = plt.subplots(1, figsize=(6,6))
        ax.hist2d(energies[unused[:ntest],1], orig_pred_scaled[:,1].flatten(),
                  bins=1000, # Just for Tests; Mila wrote 1000
                  cmin=1,  # Just for Tests; Mila wrote 1 
                  norm=mcolors.PowerNorm(0.5))
        b,t = get_limits([energies[unused[:ntest],1], hp_pred_scaled[:,1]])
        ax.set_xlabel("reference")
        ax.set_ylabel("prediction")
        ax.set_aspect("equal")
        ax.set_ylim((b,t))
        opti_ref = np.linspace(b,t,num=10)
        ax.plot(opti_ref, opti_ref, c="C1")
        fig.savefig(join(origmod_outpath, "scatter_osc.png"), dpi=300)
        plt.clf()

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
if delete_tunertrials:
    shutil.rmtree(hp_outpath) # = bash's rm -rf
    


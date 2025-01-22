import keras_tuner as kt
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from itertools import combinations

from exsNN.layers.gradients import EnergyGradientLayer
from exsNN.utils.loss import custom_loss_forces
from exsNN.esp_nn import ScaledMeanAbsoluteError
from exsNN.layers.mlp import MLP
from exsNN.scaler.general import ScalingLayer
from exsNN.layers.normalize import NormalizationLayer
from exsNN.layers.features import InverseDistance_with_ESP
from exsNN.layers.normalize import ConstLayerNormalization
from exsNN.layers.features import FeatureGeometric

class hpModelBuilder_forces:
    """
    Class for HP search for force NN.
    """
    def __init__(self, hp_dict: dict, n_atoms: int, x_train: np.ndarray):
        """
        Initialize the hyperparameters and preprocess the training data.

        Parameters:
        hp_dict (dict): Hyperparameters dictionary.
        n_atoms (int): Number of atoms.
        x_train (np.ndarray): Training data.
        """
        self.hp_dict = hp_dict
        self.n_atoms = n_atoms

        # The init method is used to precompute the normalization parameters for the inverse distance layer
        # And the normalization layer. The inverse distance layer is the first layer of the model.

        # Initialize inverse distance and filtering layer
        self.preprocessor = InverseDistance_with_ESP(x_train) # initialize inverse distance layer (set mask)
        xtrain_dist = self.preprocessor(x_train)
        dist_shape = np.shape(xtrain_dist)
        
        x_train_dist_mean = np.mean(xtrain_dist, axis=0)

        # Calculate different means for inverse distances and ESP
        dist_mean = np.mean(x_train_dist_mean[:-n_atoms])
        esp_mean = np.mean(x_train_dist_mean[-n_atoms:])

        # Initialize normalization mean
        norm_mean = np.ones(dist_shape[1])
        norm_mean[:-n_atoms] *= dist_mean
        norm_mean[-n_atoms:] *= esp_mean

        x_train_dist_var = np.var(xtrain_dist, axis=0)

        # Calculate different variances for inverse distances and ESP
        dist_var = np.mean(x_train_dist_var[:-n_atoms])
        esp_var = np.mean(x_train_dist_var[-n_atoms:])

        # Initialize normalization variance
        norm_var = np.ones(dist_shape[1])
        norm_var[:-n_atoms] *= dist_var
        norm_var[-n_atoms:] *= esp_var

        # Store normalizer
        self.normalizer = NormalizationLayer(norm_mean, norm_var)


    def build_model(self, hp):
        # Define the model
        inputs = keras.Input(shape=(self.n_atoms, 4,))
        l2_penalty = hp.Choice("l2_penalty", self.hp_dict["l2_penalty"])
        initial_lr = hp.Choice("initial_lr", self.hp_dict["initial_lr"])
        neurons = hp.Int("neurons", self.hp_dict["neurons_min"], self.hp_dict["neurons_max"], self.hp_dict["neurons_step"])
        prepped = self.preprocessor(inputs)
        normed = self.normalizer(prepped)

        # Build the model
        x1 = layers.Dense(neurons, activation='elu', kernel_regularizer=keras.regularizers.l2(l2_penalty))(normed)
        nrlayers = hp.Int("layers", self.hp_dict["layers_min"], self.hp_dict["layers_max"], self.hp_dict["layers_step"])
        for i in range(nrlayers - 1):
            x1 = layers.Dense(neurons, activation='elu', kernel_regularizer=keras.regularizers.l2(l2_penalty))(x1)
        outputs = layers.Dense(1)(x1)  # This should output energy

        # Compile the model
        loss_ratio = hp.Choice("loss_ratio", self.hp_dict["loss_ratio"])
        model = EnergyGradientLayer(inputs=inputs, outputs=outputs, n_atoms=self.n_atoms)
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(initial_lr, 1e4, t_mul=1.5, m_mul=0.3, alpha=2e-3)
        opt = keras.optimizers.Adam(lr_schedule)  # initialize optimizer
        my_loss_fn = custom_loss_forces(loss_ratio)
        model.compile(optimizer=opt, loss=my_loss_fn, metrics=["mae"])
        return model

    def perform_hp_search(self, x_train, y_train_scaled, hp_epochs, hp_factor, batch_size, stop_early):
        tuner = kt.Hyperband(self.build_model, objective="val_mae", max_epochs=hp_epochs, factor=hp_factor, hyperband_iterations=1, directory="trials")
        tuner.search(x_train, y_train_scaled, batch_size=batch_size, epochs=hp_epochs, callbacks=[stop_early], verbose=2, validation_split=0.2)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps, tuner


class hpModelBuilder_energy_oscStr:
    """
    Class for HP search for energy + oscillator strength NN.
    """
    def __init__(self, hp_dict, natoms, dense_activ, final_activ, output_spec, loss, r2_metric, norm, coords_scale_mean, coords_scale_std, feat_coords_mean, feat_coords_std):
        self.hp_dict = hp_dict
        self.natoms = natoms
        self.dense_activ = dense_activ
        self.final_activ = final_activ
        self.output_spec = output_spec
        self.loss = loss
        self.r2_metric = r2_metric
        self.norm = norm
        self.coords_scale_mean = coords_scale_mean
        self.coords_scale_std = coords_scale_std
        self.feat_coords_mean = feat_coords_mean
        self.feat_coords_std = feat_coords_std

    def build_model(self, hp):
        """ Building up the esp-model for hyperparametersearch. This is a modified 
        version of esp_nn.py/build_model() with hyperparameters. Therefore, the
        main architecture is fixed. You may vary number of neurons in MLP layer,
        the depth, the l1 or l2 regularization and the learning rate for now."""

        # 0. create interatomic distance matrix
        geom_idx = [(i,j) for i,j in combinations(range(self.natoms), 2)]
        interatomic_dists = np.array(geom_idx) # this is [ [0,1],[0,2],...,[83,84] ]
        # geom_in, geom_prep = build_geom_preprocess_layer(self.natoms, interatomic_dists)

        # 1. Create .....
        geom_shape = (self.natoms, 3)
        geom_in = keras.Input(shape=geom_shape, dtype='float32', name='geo_input')

        # Add scaling layer
        geom_in = ScalingLayer(self.coords_scale_mean, self.coords_scale_std)(geom_in)

        # feature calculation layer
        feat_layer = FeatureGeometric(invd_shape=interatomic_dists.shape, name="feat_layer")
        
        # which interatomic distances to use
        feat_layer.set_mol_index(interatomic_dists, None, None)

        # make 1D
        full = keras.layers.Flatten(name='feat_flat')(geom_in)
        full = feat_layer(geom_in)
        
        # normalization of features: norm over entire training set, using set_const_normalization_from_features
        geom_prep = ConstLayerNormalization(name="feat_std")(full)


        # 2. Esp_in
        esp_in = keras.Input(shape=(self.natoms, ), dtype='float32', name='esp_input')

        # 3. Concat
        rep = keras.layers.Concatenate(name="concat_layer")([geom_prep, esp_in]) # representation of inputs

        # 4. MLP with HP search
        neurons = hp.Int("nn_size", self.hp_dict["neurons_min"], self.hp_dict["neurons_max"], self.hp_dict["neurons_step"])
        hp_layer_depth = hp.Int("depth", self.hp_dict["layers_min"], self.hp_dict["layers_max"], self.hp_dict["layers_step"])
        hp_regularizer = self.hp_dict["regulizer"]
        mlp = MLP(dense_units=neurons,
                    dense_depth=hp_layer_depth,
                    dense_activ=self.dense_activ,
                    dense_activ_last=self.dense_activ,
                    dense_kernel_regularizer=hp_regularizer,
                    name="monolith")
        
        # 5. Output layer
        final_layer = keras.layers.Dense(2, 
                        activation=self.final_activ, 
                        use_bias=True, 
                        name="out_vom_mlp")(mlp(rep))
        
        # 6. Model
        inputs_list = [geom_in, esp_in] # actual inputs
        model = keras.Model(inputs=inputs_list, outputs=final_layer) # maybe [final_layer]?
        hp_learning_rate = hp.Choice("learning_rate", values=self.hp_dict["learning_rates"])
        opti = keras.optimizers.Adam(learning_rate=hp_learning_rate)

        # 7. Metrics
        for _, o in self.output_spec.items():
            # if targets are scaled, the MAE must be converted to original data units
            maes = []
            if o.scaler:
                mae_scaled = ScaledMeanAbsoluteError(scaling_shape=o.scaler.scale_.shape)
                mae_scaled.set_scale(o.scaler.scale_)
                maes.append(mae_scaled)
            else:
                maes.append("mean_absolute_error")
                
        # 8. Compile model
        model.compile(optimizer=opti, loss=self.loss, metrics=[[mae, self.r2_metric] for mae in maes]) # for history and tuning

        # 9. Set normalization parameters
        # # The layer calculating inverse distances ist the first layer of the model. The result of this layer is used to fit the normalization layer (x -> x-µ/std).
        # # Basically a normalization layer is fitted to the inverse distances.
        if self.feat_coords_mean is not None and self.feat_coords_std is not None:
            model.get_layer('feat_std').set_weights([self.feat_coords_mean, self.feat_coords_std])
        return model
    
    def perform_hp_search(self, x_train, y_train, hp_maxepochs, hp_factor, callbacks, hp_outpath):
        tuner = kt.Hyperband(self.build_model, objective=kt.Objective("val_r2_metric", "max"), max_epochs=hp_maxepochs, factor=hp_factor, directory=hp_outpath)
        tuner.search(x=x_train, y=y_train, verbose=2, epochs=hp_maxepochs, validation_split=0.1, callbacks=callbacks, batch_size=64)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps, tuner
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pyNNsMD.layers.gradients import EnergyGradientLayer
from pyNNsMD.utils.loss import custom_loss_forces

class hpModelBuilder:
    """
    Class for HP search for force NN.
    """
    def __init__(self, hp_dict, n_atoms, preprocessor, normalizer):
        self.hp_dict = hp_dict
        self.n_atoms = n_atoms
        self.preprocessor = preprocessor
        self.normalizer = normalizer

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
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from exsNN.layers.mlp import MLP
from exsNN.layers.normalize import ConstLayerNormalization
from exsNN.layers.features import FeatureGeometric
from exsNN.utils.loss import r2_metric
from exsNN.nn_pes_src.device import set_gpu

from sklearn.preprocessing import StandardScaler

class RoughStandardScaler(StandardScaler):
    """
        Scales array according to overall mean and std, *not* feature-wise.
        Can be used to center/scale xyz positions before feeding them to the network.
        This should actually be irrelevant for the model's training performance,
        but I'm keeping it around because it's used this way in pyNNsMD.
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)      
    def transform(self, x=None, y=None):
        x_res = x
        if x is not None:
            x_res = (x - self.mean_) / self.scale_
        return x_res
    def inverse_transform(self, x, copy):
        x_res = x
        if x is not None:
            x_res = x * self.scale_ + self.mean_
        return x_res
    def fit(self, x=None, y=None):
        npeps = np.finfo(float).eps
        self.mean_ = np.mean(x)
        self.var_ = np.var(x) + npeps
        self.scale_ = np.sqrt(self.var_)
    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y=y)


class DummyStandardScaler(StandardScaler):
    """
        Pretends to scale data as a StandardScaler would do, but uses a fixed 
        mean of 0 and variance of 1.
    """
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None, sample_weight=None):
        self.n_features_in_ = X.shape[-1]
        self.mean_ = np.zeros_like(X)
        self.var_  = np.ones_like(X)
        self.scale_ = np.sqrt(self.var_)
        return self
    


class ScaledMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
    """
        Scaled MAE to be used as keras metric when targets are scaled to normal dist.
        From pyNNsMD package.
    """

    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape, initializer=tf.keras.initializers.Ones(), name='scale_mae',
                                     dtype=tf.keras.backend.floatx())
        self.scaling_shape = scaling_shape

    def reset_state(self):
            # Super variables
            ks.backend.set_value(self.total, 0)
            ks.backend.set_value(self.count, 0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * y_true
        y_pred = self.scale * y_pred
        return super(ScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        mae_conf = super(ScaledMeanAbsoluteError, self).get_config()
        mae_conf.update({"scaling_shape": self.scaling_shape})
        return mae_conf

    def set_scale(self,scale):
        ks.backend.set_value(self.scale, scale)

class SubNet:
    """
        Dummy class to carry sub-network configuration parameters.
        They are passed as keyword arguments to the constructor and easily
        accessed during network construction.
        Which kwargs are necessary/meaningful depends is decided in the 
        specific NN construction function (e.g. build_model in this file)
        
        Examples
        --------
        >>> geom_net = SubNet(neurons = 100, depth = 3, activ = 'sigmoid')
        >>> print(geom_net.depth)
        3
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class OutputSpec:
    """
        Dummy class to carry network output configurations.
        Configuration parameters are passed as keyword arguments to the 
        constructor and easily accessed during network construction.
        Which kwargs are necessary/meaningful depends is decided in the 
        specific NN construction function (e.g. build_model in this file)
        
        Examples
        --------
        >>> output = OutputSpec(from_subnets = "geom_net")
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def precompute_feature_in_chunks(x, model, batch_size):
    """
        Precompute representations for an entire data set to speed up training.
        From pyNNsMD package.
        Returns precomputed representations as array.
    """
    np_x = []
    for i in range(int(np.ceil(len(x) / batch_size))):
        a = int(batch_size * i)
        b = int(batch_size * i + batch_size)
        tf_x = tf.convert_to_tensor(x[a:b], dtype=tf.float32)
        feat_pred = model.get_layer("feat_layer")(tf_x, training=False)
        np_x.append(np.array(feat_pred.numpy()))
    np_x = np.concatenate(np_x, axis=0)
    return np_x

def set_const_normalization_from_features(np_x, model):
    """
        Use precomputed features to set scaling parameters for inverse distances 
        across entire data set.
        From pyNNsMD package.
    """
    feat_x_mean, feat_x_std = model.get_layer('feat_std').get_weights()
    feat_x_mean = np.mean(np_x, axis=0, keepdims=True)
    feat_x_std = np.std(np_x, axis=0, keepdims=True)
    model.get_layer('feat_std').set_weights([feat_x_mean, feat_x_std])

def build_geom_preprocess_layer(natoms, interatomic_dists, norm="const"):
    """
        Prepare geometry pre-processing part of the NN:
        atomic positions -> inverse distances -> normalization

        Returns the very first input layer and the layer carrying the fully
        pre-processed representation.

        Parameters
        ----------
        natoms : int
            number of atoms in each input structure.

        interatomic_dists : 2D np.array of ints
            pairs of atomic indices whose distances to include in the representation.

        norm : str, default: "const"
            either "const" (normalize features over entire data set)
            or "batch" (use ks.layers.BatchNormalization).
    """

    geom_shape = (natoms, 3)
    geom_in = ks.Input(shape=geom_shape, dtype='float32', name='geo_input')
    # feature calculation layer
    feat_layer = FeatureGeometric(invd_shape = interatomic_dists.shape, 
                                  name="feat_layer")
    # which interatomic distances to use
    feat_layer.set_mol_index(interatomic_dists, None, None)
    # make 1D
    full = ks.layers.Flatten(name='feat_flat')(geom_in)
    full = feat_layer(geom_in)
    
    # normalization of features
    if norm == "const":
        # norm over entire training set, using set_const_normalization_from_features
        full = ConstLayerNormalization(name="feat_std")(full)
    elif norm == "batch":
        # norm over mini-batches only
        full = ks.layers.BatchNormalization(name="feat_std")(full)
    else:
        raise NotImplementedError

    return geom_in, full

def build_model(subnets, 
                output_specs,
                natoms, interatomic_dists, 
                norm="const", loss="mean_squared_error",
                learning_rate = 5e-4, 
                print_summary=True, 
                make_subnet_predictions_accessible = False):

    """
        A generic constructor for NNs. 

        Parameters
        ----------
        subnets : dict of subnet names mapping to SubNets
            SubNet objects specifying the architecture for each MLP subunit.
            This function requires that each SubNet specifies the following fields:
            - neurons (int)
                number of neurons per hidden dense layer
            - depth (int)
                number of fully connected hidden layers
            - activ (str, dict or callable)
                activation function for hidden layers
            - final_activ (str, dict or callable)
                activation function for final subnet output layer (usually "linear")
            - num_outputs (int)
                number of output neurons of the MLP subunit
            - rep (str)
                Type of input and representation. 
                "geom" assumes an input layer of shape (natoms, 3) and does
                the preprocessing steps in build_geom_preprocess_layer.
                "esp" assumes an input layer of shape (natoms, ) and does no further 
                pre-processsing.
                "both" uses both input types and concatenates the geom pre-processing
                result with the esp layer.
        
        output_specs : dict of output names mapping to OutputSpecs
            OutputSpec objects specifying the desired outputs.
            This function requires that each OutputSpec specifies the following fields:
            - from_subnets (str or list of str)
                name(s) of MLP subnets from which to collect final layer outputs
            - merge_method (ks.layers.Layer)
                If multiple subnets are passed in from_subnets, the way to aggregate 
                their outputs must be specified here in form of a valid keras layer
                name. Options are e.g: ks.layers.Add, ks.layers.Concatenate...
            - scaler (sklearn.StandardScaler) 
                Scaler for the targets. Used to convert the MAE to its "natural" scaling 
                for easy interpretability.
                If no scaling is desired for this output, use a DummyStandardScaler.
                Using a StandardScaler with e.g. `with_mean=False` will not work!

        natoms : int
            Number of atoms per input structure.
        
        interatomic_dists : 2D np.array of ints
            Pairs of atomic indices whose distances to include in the representation.

        norm : str, default="const"
            Either "const" (normalize features over entire data set)
            or "batch" (use ks.layers.BatchNormalization).

        loss : str or callable (based on ks.losses.Loss) or list of such, default="mean_squared_error"
            Loss function for model fit. When using multiple outputs, 
            multiple losses may also be used.
        
        learning_rate : float, default=5e-4
            Learning rate for the optimizer.

        print_summary : bool, default=True
            If True, model.summary() is printed to stdout.
        
        make_subnet_predictions_accessible : bool, default=False
            If True, a second model is constructed which allows the user access
            to the predictions of the individual subnets.
            This only makes limited sense when targets are scaled!


    """

    assert len(subnets) > 0, "No subnet configurations passed!"
    assert len(output_specs) > 0, "No output configurations passed!"

    # Handy shortcut for building the layers which transform
    # coordinates -> inverse distances -> feature-wise normalized inverse distances
    geom_in, geom_prep = build_geom_preprocess_layer(natoms, 
                                                     interatomic_dists, 
                                                     norm=norm)
    
    # The ESP input layer is simpler
    esp_in = ks.Input(shape=(natoms,), dtype='float32', name='esp_input')

    # flags for whether to expect geometry or esp inputs:
    any_geom, any_esp = False, False

    # construct MLP-type sub-networks according to config specified in subnets variable. 
    # Each subnet specifies whether to use geometry or ESP as input.
    mlps = {}
    final_layers = {}
    for name, subnet in subnets.items():
        print(f"adding subnet {name}")
        mlps[name] = MLP(subnet.neurons, 
                         dense_depth=subnet.depth, 
                         dense_bias=True,
                         dense_activ=subnet.activ,
                         name=name)
        if subnet.rep == "esp":
            rep = esp_in
            any_esp = True
        elif subnet.rep == "geom":
            rep = geom_prep
            any_geom = True
        elif subnet.rep == "both":
            any_esp, any_geom =  True, True
            #rep = ks.layers.Concatenate()([geom_prep, esp_in])
            rep = ks.layers.Concatenate(name="concat_layer")([geom_prep, esp_in])
        else:
            raise NotImplementedError(f"The representation {subnet.rep} is currently not supported. Use one of 'esp' (shape=(natoms,), no preprocessing), 'geom' (shape=(natoms, 3), featurization& normalization), or 'both' (concatenates the 'esp' and 'geom' representations)!")

        res = mlps[name](rep)
        final_layers[name] = ks.layers.Dense(subnet.num_outputs, 
                                             activation=subnet.final_activ, 
                                             use_bias=True, 
                                             name=f"out_{name}")(res)

    # use flags obtained from subnet specs to define NN input layers
    inputs = []
    if any_geom:
        inputs.append(geom_in)
    if any_esp:
        inputs.append(esp_in)
    assert len(inputs)>0, "Something went wrong and neither geometry nor ESP inputs are used!"

    # parse outputspecs to construct outputs
    outputs = []
    maes = []
    scaling_flag = False
    for name, o in output_specs.items(): # name ist "QM/MM energy" und o ist 
    #esp_nn.OutputSpec mit o[from_subnets]="monolith" und o[scaler]=StandardScaler
        if isinstance(o.from_subnets, list): # nein, ist str 'monolith'
            out = o.merge_method()([final_layers[i] for i in o.from_subnets])
        else:
            out = final_layers[o.from_subnets]
        outputs.append(out)

        # if targets are scaled, the MAE must be converted to original data units
        if o.scaler: # ja
            mae_scaled = ScaledMeanAbsoluteError(scaling_shape=o.scaler.scale_.shape)
            mae_scaled.set_scale(o.scaler.scale_)
            maes.append(mae_scaled)
            if o.scaler.scale_.any() != 1: # ja weil oben gescaled wurde
                scaling_flag = True
        else:
            maes.append("mean_absolute_error")

    # make a model out of it all
    model = ks.Model(inputs=inputs, outputs=outputs)
    opti = ks.optimizers.Adam(learning_rate=learning_rate)

    # final model configuration
    model.compile(optimizer=opti, 
                  loss = loss,
                  metrics=[[mae, r2_metric] for mae in maes])

    if print_summary:
        print(model.summary())

    # if desired, you can also get the subnets' outputs for analysis.
    # for this, build a second model which outputs the individual subnet outputs. 
    if make_subnet_predictions_accessible:
        subnet_output = ks.Model(inputs=inputs, outputs=[l for l in final_layers.values()])
        if scaling_flag:
            print("WARNING: Some targets have been scaled to normal distributions, limiting subnet output interpretability!")
        return model, subnet_output 
    else: 
        return model

def get_limits(data):
    """ 
        Return common upper and lower bounds for a number of arrays 
        (used for plotting).
    """
    t = max([i.max() for i in data])
    b = min([i.min() for i in data])
    return b,t

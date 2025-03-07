import tensorflow as tf
import tensorflow.keras as ks

class WrapForcesModel(ks.Model):
	#important to save model for MLMM
	"""
	Wraps the NN model for the forces.
	Neccessary to store model for MLMM with GROMACS.
    Input: xyz in Bohr and ESP in Hartree
    Output: Forces in Hartree/Bohr, Energie in Hartree
    """
	def __init__(self, model, mean, var):
		super().__init__()
		self.submodel = model
		# self.mean = mean
		# self.std = tf.sqrt(var)
		self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
		self.std = tf.sqrt(tf.convert_to_tensor(var, dtype=tf.float32))
	def call(self, inputs):
		tf.print("New call Wrapper\nScaling: ", self.mean, self.std, self.mean.dtype, self.std.dtype)
		tf.print("Inputs Wrapper:\n", inputs, inputs.dtype, type(inputs), summarize=-1)
		outputs = self.submodel(inputs)
		tf.print("Outputs Wrapper ", outputs.dtype, type(outputs))
		tf.print("dtypes: ", self.mean.dtype, self.std.dtype, outputs.dtype)
        # Convert outputs to float64
		outputs = tf.convert_to_tensor(outputs, tf.float32)
		tf.print("After Conversion")
    
		# Check for NaNs or Infs in outputs
		tf.debugging.check_numerics(outputs, "NaNs or Infs found in outputs")

		outputs_rescaled = self.std * outputs + self.mean
		tf.print("Rescaled Outputs Successfully Calculated")
		tf.print(outputs_rescaled.dtype, type(outputs_rescaled))
		tf.print("Outputs Rescaled Wrapper:\n", outputs_rescaled, summarize=-1)
		return outputs_rescaled

	def get_config(self):
		config = super().get_config().copy()
		return config

class WrapEnergyModel(ks.Model):
    """
    Wraps the NN model for the energies + oscillator strengths.
    Input: xyz in Bohr and ESP in Hartree
    Output: Unit of energy in Hartree
    """
    def __init__(self, model, mean, var):
        super().__init__()
        self.submodel = model
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std = tf.sqrt(tf.convert_to_tensor(var, dtype=tf.float32))

    def call(self, inputs):
        outputs = self.submodel(inputs)
        outputs_rescaled = self.std * outputs + self.mean
        return outputs_rescaled
    
    def get_config(self):
        config = super().get_config().copy()
        return config
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
		# tf.print(">> INPUT INFO")
		# tf.print("Shape Inputs Wrapper:", inputs.shape)
		# tf.print("New call Wrapper\nScaling: ", self.mean, self.std, self.mean.dtype, self.std.dtype)
		# tf.print("Inputs Wrapper:\n", inputs, summarize=-1) ###########
		# tf.print("Inputs Wraper types :", inputs.dtype, type(inputs))
		outputs = self.submodel(inputs)
		# tf.print(">> OUTPUT INFO")
		# tf.print("Outputs Wrapper types ", outputs.dtype, type(outputs))
		# tf.print("self dtypes: ", self.mean.dtype, self.std.dtype, outputs.dtype)


        # Convert outputs to float32
		outputs = tf.convert_to_tensor(outputs, tf.float32)
		# tf.print(">> Conversion successfull")
    
		# Check for NaNs or Infs in outputs
		# tf.debugging.check_numerics(outputs, "NaNs or Infs found in outputs")

		outputs_rescaled = self.std * outputs + self.mean
		# tf.print(">> Outputs rescaled successfull")
		# tf.print("Shape of outputs_rescaled:", outputs_rescaled.shape)

		# Try printing a small sample instead of the whole tensor
		# tf.print("First few values:", outputs_rescaled[0, :5])

		# Add debug code here
		# tf.print(">> DEBUGGING INFO:")
		# tf.print("Min value:", tf.reduce_min(outputs))
		# tf.print("Min value (rescaled):", tf.reduce_min(outputs_rescaled))

		# tf.print("Max value:", tf.reduce_max(outputs))
		# tf.print("Max value (rescaled):", tf.reduce_max(outputs_rescaled))

		# tf.print("Contains NaN:", tf.reduce_any(tf.math.is_nan(outputs)))
		# tf.print("Contains NaN (rescaled):", tf.reduce_any(tf.math.is_nan(outputs_rescaled)))

		# tf.print("Contains Inf:", tf.reduce_any(tf.math.is_inf(outputs)))
		# tf.print("Contains Inf (rescaled):", tf.reduce_any(tf.math.is_inf(outputs_rescaled)))

		# tf.print("Rescaled Outputs Successfully Calculated")
		# tf.print("Outputs Wrapper types ", outputs.dtype, type(outputs))
		# tf.print("Outputs Wrapper types (rescaled) ", outputs_rescaled.dtype, type(outputs_rescaled))

		# # Masks
		# try:
		# 	tf.print(">> NaNs and Infs in outputs:")
		# 	NaN_mask = tf.reduce_any(tf.math.is_nan(outputs), axis=1)
		# 	tf.print("NaNs:\n")
		# 	tf.print(tf.boolean_mask(outputs, NaN_mask))
		# 	Inf_mask = tf.reduce_any(tf.math.is_inf(outputs), axis=1)
		# 	tf.print("Infs:\n")
		# 	tf.print(tf.boolean_mask(outputs, Inf_mask))
		# except:
		# 	tf.print("-----> Could not print NaNs and Infs in outputs")

		# # Rescaled mask
		# try:
		# 	tf.print(">> NaNs and Infs in rescaled outputs:")
		# 	NaN_mask = tf.reduce_any(tf.math.is_nan(outputs_rescaled), axis=1)
		# 	tf.print("NaNs:\n")
		# 	tf.print(tf.boolean_mask(outputs_rescaled, NaN_mask))
		# 	Inf_mask = tf.reduce_any(tf.math.is_inf(outputs_rescaled), axis=1)
		# 	tf.print("Infs:\n")
		# 	tf.print(tf.boolean_mask(outputs_rescaled, Inf_mask))
		# except:
		# 	tf.print("-----> Could not print NaNs and Infs in rescaled outputs")

		# try:
		# 	tf.print("Outputs Wrapper:", outputs, summarize=-1)
		# 	tf.print("Outputs Rescaled Wrapper:\n", outputs_rescaled, summarize=-1) #######
		# 	tf.print("Successfully printed outputs")
		# except:
		# 	tf.print("-----> Outputs Rescaled Wrapper: Could not print")
		
		# tf.print("----------------------------------------------------------------------------------------")
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
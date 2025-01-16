import tensorflow as tf
import tensorflow.keras as ks

class WrapForcesModel(ks.Model):
	#important to save model for MLMM
	"""
	Wraps the NN model for the forces.
	Neccessary to store model for MLMM with GROMACS.
    """
	def __init__(self, model, mean, var):
		super().__init__()
		self.submodel = model
		self.mean = mean
		self.std = tf.sqrt(var)
	def call(self, inputs):
		outputs = self.submodel(inputs)
		outputs_rescaled = self.std * outputs + self.mean
		tf.print("wrap", outputs.shape)
		return outputs_rescaled
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


class InverseDistance(ks.layers.Layer): #taken from Milas code, added ESP and mask
	def __init__(self,mask1):
		super(InverseDistance,self).__init__()
		self.mask1 = mask1	#this mask filters large distances and reduces the inpu the dimension
	def build(self,input_shape):
		super(InverseDistance,self).build(input_shape)
	def call(self,inputs):
		coords = inputs[:,:,:3]
		esp = inputs[:,:,3]
		ins_int = K.int_shape(coords)
		ins = K.shape(coords)
		a = K.expand_dims(coords,axis=1)
		b = K.expand_dims(coords,axis=2)
		c = b-a
		d = K.sum(K.square(c),axis=-1)
		ind1 = K.expand_dims(K.arange(0,ins_int[1]),axis=1)
		ind2 = K.expand_dims(K.arange(0,ins_int[1]),axis=0)
		mask = K.less(ind1,ind2)
		mask = K.expand_dims(mask,axis=0)
		mask = K.tile(mask,(ins[0],1,1))
		d = d[mask]
		d = K.reshape(d,(ins[0],(ins_int[1]*(ins_int[1]-1))//2))
		d = K.sqrt(d)
		out = 1/d
		out = tf.transpose(tf.boolean_mask(tf.transpose(out),self.mask1))	#comment to turn off filtering
		out = K.concatenate((out,esp),axis=-1)
		return out

class FirstInverseDistance(ks.layers.Layer): #to generate mask1 that filters large distances from input
	def __init__(self):
		super(FirstInverseDistance,self).__init__()
	def build(self,input_shape):
		super(FirstInverseDistance,self).build(input_shape)
	def call(self,inputs):
		coords = inputs[:,:,:3]
		esp = inputs[:,:,3]
		ins_int = K.int_shape(coords)
		ins = K.shape(coords)
		a = K.expand_dims(coords,axis=1)
		b = K.expand_dims(coords,axis=2)
		c = b-a
		d = K.sum(K.square(c),axis=-1)
		ind1 = K.expand_dims(K.arange(0,ins_int[1]),axis=1)
		ind2 = K.expand_dims(K.arange(0,ins_int[1]),axis=0)
		mask = K.less(ind1,ind2)
		mask = K.expand_dims(mask,axis=0)
		mask = K.tile(mask,(ins[0],1,1))
		d = d[mask]
		d = K.reshape(d,(ins[0],(ins_int[1]*(ins_int[1]-1))//2))
		d = K.sqrt(d)
		out = 1/d
		mask1 = K.greater(out,0.1)	#this checks if the inverse distance of a pair of atoms is too low
		#mask1=K.less(out,0.1)
		#mask1=tf.math.logical_not(mask1)	#this checks if the inverse distance of a pair of atoms is high enough
		out = K.concatenate((out,esp),axis=-1)
		return out, mask1
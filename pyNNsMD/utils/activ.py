"""
Smooth activation functions for tensorflow.keras.
"""

import tensorflow.keras as ks


def leaky_softplus(x, alpha=0.3):
    """
    Leaky softplus activation function similar to leakyRELU but smooth.
        
    Args:
        alpha (float, optional): Leaking slope. The default is 0.3.

    Returns:
        output of function of x.
    """
    def activation(x):
        return ks.activations.softplus(x) * (1 - alpha) + alpha * x
    return activation(x)


def shifted_softplus(x):
    """
    Softplus function from tf.keras shifted downwards.

    Args:
        x (tf.tensor): Activation input.

    Returns:
        tf.tensor: Activation.

    """
    return ks.activations.softplus(x) - ks.backend.log(2.0)

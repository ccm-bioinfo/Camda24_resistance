import numpy as np
from sklearn.pipeline import FunctionTransformer

def round_log2(x):
    """Rounding of log2 transformation."""
    return np.round(np.log2(x))

def exp2(x):
    """Exponential transformation."""
    return np.exp2(x)

def reshape(x):
    """"Reshape the input array to a 2D array of shape (-1, 1)."""
    return x.values.reshape(-1, 1)

def dimension_printer(x):
    """Print the shape of the input."""
    print(f"Object of {type(x)} has shape: {x.shape}")
    return x



Log2Transformer = FunctionTransformer(func=np.log, validate=True)

RoundTransformer = FunctionTransformer(func=np.round, validate=True)

Exp2Transformer = FunctionTransformer(func=exp2, validate=True)

ReshapeTransformer = FunctionTransformer(reshape, validate=True)
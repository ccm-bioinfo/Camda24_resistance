import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer

def round_log2(x):
    """Rounding of log2 transformation."""
    return np.round(np.log2(x))

def exp2(x):
    """Exponential transformation."""
    return np.exp2(x)

def dimension_printer(x):
    """Print the shape of the input."""
    print(f"Object of {type(x)} has shape: {x.shape}")
    return x

class ReshapeTransformer(BaseEstimator, TransformerMixin):
    """Transformer to reshape input data to (-1, 1) and store original dimensions."""
    def __init__(self):
        self.original_shape = None

    def fit(self, X, y=None):
        # No fitting required, just return self
        return self

    def transform(self, X):
        # Store the original shape
        self.original_shape = X.shape
        return X.reshape(-1, 1)

    def inverse_transform(self, X):
        if self.original_shape is None:
            raise ValueError("Original shape is not defined. You need to call transform first.")
        # Restore the original shape
        return X.reshape(self.original_shape)

Log2Transformer = FunctionTransformer(func=np.log, validate=True)

RoundTransformer = FunctionTransformer(func=np.round, validate=True)

Exp2Transformer = FunctionTransformer(func=exp2, validate=True)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from common.config import DISCRETE_LIMIT

class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, discrete_limit=DISCRETE_LIMIT, category_transformer=None, continuous_transformer=None):
        self.discrete_limit = discrete_limit
        self.category_transformer = category_transformer
        self.continuous_transformer = continuous_transformer
        self.categorical_cols = None
        self.continuous_cols = None
        self.original_index = None
        self.cardinalities = None
        self.column_transformer = None

    def _init_column_transformer(self):
        """Initialize the column transformer with appropriate transformers."""
        if self.continuous_transformer is None:
            self.continuous_transformer = StandardScaler()  # Default continuous transformer (e.g., standardization)
        if self.category_transformer is None:
            self.category_transformer = OneHotEncoder(handle_unknown='ignore')  # Default categorical transformer
        # Initialize the column transformer
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('category_transformer', self.category_transformer, self.categorical_cols),
                ('continuous_transformer', self.continuous_transformer, self.continuous_cols)
            ], remainder='drop'
        )

    def fit(self, X):
        """Fit the transformer to the data, identifying categorical and continuous columns."""
        self.categorical_cols = [col for col in X.columns if X[col].nunique() <= self.discrete_limit]
        self.continuous_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > self.discrete_limit]
        self.original_index = X.columns
        self._init_column_transformer()
        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        """Apply the transformations (encoding and passthrough) to the data."""
        return self.column_transformer.transform(X)

def inverse_transform(self, X_transformed: np.ndarray) -> pd.DataFrame:
    """Inverse transform the transformed data back to the original feature space."""
    if self.column_transformer is None or not hasattr(self.column_transformer, 'transformers_'):
        raise ValueError("The transformer is not fitted yet. Please fit the transformer before inverse transforming the data.")

    # Use the column_transformer's inverse_transform method
    X_inverse_transformed = self.column_transformer.inverse_transform(X_transformed)

    # Create a DataFrame with the original column names
    result = pd.DataFrame(X_inverse_transformed, columns=self.original_index)

    # Reindex the result to match the original index
    result = result.reindex(index=X_transformed.index)

    return result

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'feature3': [10, 20, 30, 40, 50],
        'mic': [0.5, 1.0, 2.0, 4.0, 8.0],
        'phenotype': ['resistant', 'susceptible', 'resistant', 'susceptible', 'resistant']
    })

    df.dropna(inplace=True)
    # Split the data into features and target
    X = df.loc[:, ~df.columns.isin(['mic', 'phenotype'])]
    
    # Initialize the transformer 
    transformer = FeaturesTransformer(discrete_limit=15)
    # Fit and transform the data
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    # Inverse transform the data
    X_transformed_inverse = transformer.inverse_transform(X_transformed)
    
    # Check if the original features are the same as the transformed features
    assert X_transformed_inverse.equals(X), "The original features are not the same as the transformed features"
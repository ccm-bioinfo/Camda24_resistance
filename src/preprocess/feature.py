from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


from config import DISCRETE_LIMIT

class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, discrete_limit=DISCRETE_LIMIT):
        self.discrete_limit = discrete_limit
        self.transformer = None
        self.categorical_cols = None
        self.continuous_cols = None
        self.original_index = None
        self.cardinalities = None
        self._column_transformer = ColumnTransformer(
            transformers=[
                ('onehot_categorical', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols),
                ('pass_continuous', 'passthrough', self.continuous_cols)
            ], remainder='drop'
        )

    def fit(self, X, y=None):
        self.categorical_cols = [col for col in X.columns if X[col].nunique() <= self.discrete_limit]
        self.continuous_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > self.discrete_limit]
        self.original_index = X.columns
        self.column_transformer.fit(X)
        self.cardinalities = [len(c) for c in self.column_transformer.transformers_[0][1].categories_]
        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        if self.column_transformer is None:
            raise ValueError("The transformer is not fitted yet. Please fit the transformer before inverse transforming the data.")
        current_pos = 0
        categorical_inverse = []
        for cardinality in self.cardinalities:
            categorical_data = X_transformed[:, current_pos:current_pos + cardinality]
            inverse_categorical = self.column_transformer.transformers_[0][1].inverse_transform(categorical_data)
            categorical_inverse.append(inverse_categorical)
            current_pos += cardinality
        continuous_inverse = X_transformed[:, current_pos:]
        all_inverse = np.hstack(categorical_inverse + [continuous_inverse])
        return pd.DataFrame(all_inverse, columns=self.original_index)


# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
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
    X= df.loc[:, ~df.columns.isin(['mic', 'phenotype'])]
    # Initialize the transformer 
    transformer = FeaturesTransformer(discrete_limit=15)
    # Fit and transform the data
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    # Inverse transform the data
    X_transformed_inverse = transformer.inverse_transform(X_transformed)
    # check if the original features are the same as the transformed features
    assert X_transformed_inverse.equals(X), "The original features are not the same as the transformed features"

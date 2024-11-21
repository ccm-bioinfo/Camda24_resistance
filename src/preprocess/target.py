from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from utils import Log2Transformer, RoundTransformer, ReshapeTransformer

class TargetsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, apply_multiclass_encoding=False, apply_log2=False, apply_round_to_int=False):
        self.apply_multiclass_encoding = apply_multiclass_encoding
        self.apply_log2 = apply_log2
        self.apply_round_to_int = apply_round_to_int
        self.mic_pipeline_steps = [('reshape', ReshapeTransformer)]
        self.phenotype_pipeline_steps = [('reshape', ReshapeTransformer)]
        self.mic_pipeline = None
        self.phenotype_pipeline = None

    def _add_transformers_to_pipeline(self):
        if self.apply_log2:
            self.mic_pipeline_steps.append(('Log2', Log2Transformer))
        if self.apply_round_to_int:
            self.mic_pipeline_steps.append(('Round', RoundTransformer))
        if self.apply_multiclass_encoding:
            self.mic_pipeline_steps.append(('onehot', OneHotEncoder(sparse=False)))

    def fit(self, X, y):
        self._add_transformers_to_pipeline()
        self.mic_pipeline = Pipeline(self.mic_pipeline_steps)
        self.phenotype_pipeline = Pipeline([('label_encoding', LabelEncoder()),
            ('reshape', ReshapeTransformer)
        ])
        self.mic_pipeline.fit(y["mic"])
        self.phenotype_pipeline.fit(y["phenotype"])
        return self

    def transform(self, X):
        print("Not a canonical transformer. Use fit_transform instead.")
    
    def fit_transform(self, X, y):
        self.fit(y)
        y_mic_transformed = self.mic_pipeline.transform(y["mic"].values.reshape(-1, 1))
        print(f"Mic targets transformed from shape: {y['mic'].shape} to shape: {y_mic_transformed.shape}")
        y_phenotype_transformed = self.phenotype_pipeline.transform(y["phenotype"].values.reshape(-1, 1))
        print(f"Phenotype targets transformed from shape: {y['phenotype'].shape} to shape: {y_phenotype_transformed.shape}")
        y_transformed = np.hstack((y_mic_transformed, y_phenotype_transformed))
        print('Targets transformed from shape:', y.shape, 'to shape:', y_transformed.shape)
        return y_transformed

    def inverse_transform(self, y_transformed):
        if self.mic_pipeline is None or self.phenotype_pipeline is None:
            raise ValueError("The pipeline is not fitted yet. Please call fit before inverse_transform.")
        y_mic_transformed = y_transformed[:, 0].reshape(-1, 1)
        y_phenotype_transformed = y_transformed[:, 1].reshape(-1, 1)
        y_mic_inverse = self.mic_pipeline.inverse_transform(y_mic_transformed)
        y_phenotype_inverse = self.phenotype_pipeline.inverse_transform(y_phenotype_transformed)
        y_inverse = np.hstack((y_mic_inverse, y_phenotype_inverse))
        print('Targets inverse-transformed from shape:', y_transformed.shape, 'to shape:', y_inverse.shape)
        self._verify_inverse_transform(y_transformed, y_inverse)

        return y_inverse

    def _verify_inverse_transform(self, y, y_transformed_inverse):
        mic_errors = y[:, 0] - y_transformed_inverse[:, 0]
        print("MIC transform errors statistics (min, Q1, median, Q3, max):")
        print(np.percentile(mic_errors, [0, 25, 50, 75, 100]))
        phenotype_original = y[:, 1]
        phenotype_transformed_inverse = y_transformed_inverse[:, 1]
        cm = confusion_matrix(phenotype_original, phenotype_transformed_inverse)
        print("Confusion Matrix for Phenotype:")
        print(cm)

# Example usage
if __name__ == "__main__":
    # Example data
    y = pd.DataFrame({
        "mic": [1, 2, 3, 4],
        "phenotype": ["A", "B", "A", "C"]
    })
    # Initialize transformer
    transformer = TargetsTransformer(apply_log2=True, apply_round_to_int=True)
    # Fit and transform
    y_transformed = transformer.fit_transform(None, y)
    # Inverse transform and verify accuracy
    y_transformed_inverse = transformer.inverse_transform(y_transformed)
    # Verify  the inverse transformation is accurate
    transformer._verify_inverse_transform(y, y_transformed_inverse)
    

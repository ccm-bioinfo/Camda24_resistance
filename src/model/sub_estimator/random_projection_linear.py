import numpy as np
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class RandomProjectionModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self,  regressor_params=None, classifier_params=None):
        """
        Initializes random projection and models for regression and classification.

        Args:
            n_components (int): Number of dimensions after projection.
            projection_type (str): 'gaussian' for GaussianRandomProjection or 'sparse' for SparseRandomProjection.
            random_state (int): Random seed for reproducibility.
        """
        self.regressor_n_components = regressor_params['n_components'] or  5
        self.regressor_projection_type = regressor_params['projection_type'] or 'gaussian' 
        self.classifier_n_components = classifier_params['n_components'] or 2
        self.classifier_projection_type = classifier_params['projection_type'] or 'gaussian'
        self.random_state = classifier_params['random_state'] or 42

        if projection_type == 'gaussian':
            self.random_projection = GaussianRandomProjection(
                n_components=n_components, random_state=random_state
            )
        elif projection_type == 'sparse':
            self.random_projection = SparseRandomProjection(
                n_components=n_components, random_state=random_state
            )
        else:
            raise ValueError("Invalid projection_type. Use 'gaussian' or 'sparse'.")

        self.linear_regressor = LinearRegression()
        self.logistic_classifier = LogisticRegression(penalty='none')

    def fit(self, X, y):
        """
        Fits the random projection and models with the provided data.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target matrix where the last column is for classification.
        
        Returns:
            self: Fitted model.
        """
        X_projected = self.random_projection.fit_transform(X)
        y_reg = y[:, :-1]  # All columns except last for regression
        y_clf = y[:, -1]   # Last column for classification (binary or multiclass)

        # Fit models
        self.linear_regressor.fit(X_projected, y_reg)
        self.logistic_classifier.fit(X_projected, y_clf)
        
        return self

    def predict(self, X):
        """
        Predicts regression and classification outputs for the given data.

        Args:
            X (ndarray): Feature matrix.
        
        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        X_projected = self.random_projection.transform(X)
        reg_pred = self.linear_regressor.predict(X_projected)
        clf_pred = self.logistic_classifier.predict(X_projected).reshape(-1, 1)
        return np.column_stack([reg_pred, clf_pred])

if __name__ == '__main__':
    # Test the RandomProjectionModel
    X = np.random.rand(10, 5)
    y = np.hstack([
        np.random.rand(10, 1),    # Regression target
        np.random.randint(0, 2, (10, 1))  # Classification target
    ])
    model = RandomProjectionModel(n_components=3, projection_type='gaussian', random_state=42)
    model.fit(X, y)
    print(model.predict(X))
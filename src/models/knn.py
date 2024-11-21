import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class KNNModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        """
        Initializes KNN regressor and classifier with the specified number of neighbors.
        """
        self.n_neighbors = n_neighbors
        self.knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        """
        Fits the model with the provided data.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target matrix where the last column is for classification.
        
        Returns:
            self: Fitted model.
        """
        y_reg = y[:, :-1]  # All columns except last for regression
        y_clf = y[:, -1]   # Last column for classification

        # Fit KNN models
        self.knn_regressor.fit(X, y_reg)
        self.knn_classifier.fit(X, y_clf)
        
        return self

    def predict(self, X):
        """
        Predicts regression and classification outputs for the given data.

        Args:
            X (ndarray): Feature matrix.
        
        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        reg_pred = self.knn_regressor.predict(X)
        clf_pred = self.knn_classifier.predict(X).reshape(-1, 1)
        return np.column_stack([reg_pred, clf_pred])

if __name__ == '__main__':
    # Test the KNNModel
    X = np.random.rand(10, 5)
    y = np.hstack([
        np.random.rand(10, 1),    # Regression target
        np.random.randint(0, 2, (10, 1))  # Classification target
    ])
    model = KNNModel(n_neighbors=3)
    model.fit(X, y)
    print(model.predict(X))
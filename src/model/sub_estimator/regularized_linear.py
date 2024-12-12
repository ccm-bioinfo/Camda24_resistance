import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class LogisticAndLinearModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self):
        """
        Initializes a Linear Regression model for regression and 
        a Logistic Regression model for classification.
        """
        self.linear_regressor = LinearRegression()
        self.logistic_classifier = LogisticRegression()

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
        y_clf = y[:, -1]   # Last column for classification (binary or multiclass)

        # Fit models
        self.linear_regressor.fit(X, y_reg)
        self.logistic_classifier.fit(X, y_clf)
        
        return self

    def predict(self, X):
        """
        Predicts regression and classification outputs for the given data.

        Args:
            X (ndarray): Feature matrix.
        
        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        reg_pred = self.linear_regressor.predict(X)
        clf_pred = self.logistic_classifier.predict(X).reshape(-1, 1)
        return np.column_stack([reg_pred, clf_pred])

if __name__ == '__main__':
    # Test the LogisticAndLinearModel
    X = np.random.rand(10, 5)
    y = np.hstack([
        np.random.rand(10, 1),    # Regression target
        np.random.randint(0, 2, (10, 1))  # Classification target
    ])
    model = LogisticAndLinearModel()
    model.fit(X, y)
    print(model.predict(X))
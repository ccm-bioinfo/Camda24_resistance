import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class RandomForestModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, regressor_params=None, classifier_params=None, random_state=42):
        """
        Initializes RandomForest models for regression and classification.
        """
        regressor_params['random_state'] = random_state
        classifier_params['random_state'] = random_state 
        self.rf_regressor = RandomForestRegressor(regressor_params)
        self.rf_classifier = RandomForestClassifier(classifier_params)

    def fit(self, X, y):
        """
        Fits the model with the provided data.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target matrix where the last column is for classification.
        
        Returns:
            self: Fitted model.
        """
        y_reg = y[:, :-1]  # All columns except the last for regression
        y_clf = y[:, -1]   # Last column for classification (binary or multiclass)

        # Fit models
        self.rf_regressor.fit(X, y_reg)
        self.rf_classifier.fit(X, y_clf)
        
        return self

    def predict(self, X):
        """
        Predicts regression and classification outputs for the given data.

        Args:
            X (ndarray): Feature matrix.
        
        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        reg_pred = self.rf_regressor.predict(X)
        clf_pred = self.rf_classifier.predict(X).reshape(-1, 1)
        return np.column_stack([reg_pred, clf_pred])

if __name__ == '__main__':
    # Test the RandomForestModel
    X = np.random.rand(10, 5)
    y = np.hstack([
        np.random.rand(10, 1),    # Regression target
        np.random.randint(0, 2, (10, 1))  # Classification target
    ])
    model = RandomForestModel(n_estimators=50, random_state=42)
    model.fit(X, y)
    print(model.predict(X))
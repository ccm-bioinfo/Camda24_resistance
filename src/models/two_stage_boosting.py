import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class TwoStageBoostingModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=None):
        """
        Initializes two-stage Gradient Boosting models for regression and classification.

        Args:
            n_estimators (int): Number of boosting stages to be run.
            random_state (int): Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state

        # First-stage models
        self.dominant_classifier = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        self.dominant_regressor = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)

        # Second-stage models
        self.refine_classifier = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        self.refine_regressor = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        """
        Fits the two-stage model.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target matrix where the last column is for classification.
        
        Returns:
            self: Fitted model.
        """
        y_reg = y[:, :-1].ravel()  # All columns except the last for regression
        y_clf = y[:, -1]           # Last column for classification (binary or multiclass)

        # Determine dominant class and regression value
        dominant_class = np.bincount(y_clf.astype(int)).argmax()
        dominant_indices = (y_clf == dominant_class)
        dominant_reg_value = np.median(y_reg[dominant_indices])

        self.dominant_class = dominant_class
        self.dominant_reg_value = dominant_reg_value

        # First-stage training: dominant class and value
        self.dominant_classifier.fit(X, dominant_indices)
        self.dominant_regressor.fit(X, np.full(len(X), dominant_reg_value))

        # Second-stage training: refine predictions for non-dominant cases
        refine_indices = ~dominant_indices
        if np.any(refine_indices):
            self.refine_classifier.fit(X[refine_indices], y_clf[refine_indices])
            self.refine_regressor.fit(X[refine_indices], y_reg[refine_indices])
        
        return self

    def predict(self, X):
        """
        Predicts using the two-stage approach.

        Args:
            X (ndarray): Feature matrix.
        
        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        # Stage 1: Predict dominant class and value
        dominant_class_pred = self.dominant_classifier.predict(X)
        dominant_reg_pred = self.dominant_regressor.predict(X)

        # Stage 2: Refine for non-dominant cases
        non_dominant_indices = (dominant_class_pred == 0)
        refined_class_pred = dominant_class_pred.copy()
        refined_reg_pred = dominant_reg_pred.copy()

        if np.any(non_dominant_indices):
            refined_class_pred[non_dominant_indices] = self.refine_classifier.predict(X[non_dominant_indices])
            refined_reg_pred[non_dominant_indices] = self.refine_regressor.predict(X[non_dominant_indices])

        return np.column_stack([refined_reg_pred, refined_class_pred])

if __name__ == '__main__':
    # Test the TwoStageBoostingModel
    X = np.random.rand(20, 5)
    y = np.hstack([
        np.random.rand(20, 1),    # Regression target
        np.random.randint(0, 2, (20, 1))  # Classification target
    ])
    model = TwoStageBoostingModel(n_estimators=50, random_state=42)
    model.fit(X, y)
    print(model.predict(X))
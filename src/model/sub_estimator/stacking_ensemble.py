import numpy as np
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class StackingEnsembleModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self):
        """
        Initializes Stacking models for regression and classification.
        """
        # Base estimators for regression
        reg_estimators = [
            ('lr', LinearRegression()),
            ('tree', DecisionTreeRegressor())
        ]
        # Base estimators for classification
        clf_estimators = [
            ('svc', SVC(probability=True)),
            ('tree', DecisionTreeClassifier())
        ]

        # Initialize stacking models
        self.stack_regressor = StackingRegressor(estimators=reg_estimators, final_estimator=LinearRegression())
        self.stack_classifier = StackingClassifier(estimators=clf_estimators, final_estimator=LogisticRegression())

    def fit(self, X, y):
        """
        Fits the stacking models with the provided data.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target matrix where the last column is for classification.
        
        Returns:
            self: Fitted model.
        """
        y_reg = y[:, :-1]  # All columns except the last for regression
        y_clf = y[:, -1]   # Last column for classification (binary or multiclass)

        # Fit models
        self.stack_regressor.fit(X, y_reg.ravel())  # StackingRegressor expects 1D target
        self.stack_classifier.fit(X, y_clf)
        
        return self

    def predict(self, X):
        """
        Predicts regression and classification outputs for the given data.

        Args:
            X (ndarray): Feature matrix.
        
        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        reg_pred = self.stack_regressor.predict(X).reshape(-1, 1)
        clf_pred = self.stack_classifier.predict(X).reshape(-1, 1)
        return np.column_stack([reg_pred, clf_pred])

if __name__ == '__main__':
    # Test the StackingEnsembleModel
    X = np.random.rand(10, 5)
    y = np.hstack([
        np.random.rand(10, 1),    # Regression target
        np.random.randint(0, 2, (10, 1))  # Classification target
    ])
    model = StackingEnsembleModel()
    model.fit(X, y)
    print(model.predict(X))
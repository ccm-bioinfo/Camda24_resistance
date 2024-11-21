import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class DummyModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self):
        # Dummy Regressor and Dummy Classifier
        self.dummy_regressor = DummyRegressor(strategy='median')
        self.dummy_classifier = DummyClassifier(strategy='most_frequent')

    def fit(self, X, y):
        # Separate target into regression and classification tasks
        y_reg = y[:, :-1]  # Assuming last column is binary class for classification
        y_clf = y[:, -1]  # Binary class for classification

        # Fit dummy regressor for regression task
        self.dummy_regressor.fit(X, y_reg)
        
        # Fit dummy classifier for classification task
        self.dummy_classifier.fit(X, y_clf)
        
        return self

    def predict(self, X):
        # Predict using dummy regressor for regression output
        reg_pred = self.dummy_regressor.predict(X)
        # Predict using dummy classifier for classification output
        clf_pred = self.dummy_classifier.predict(X)
        return np.column_stack([reg_pred, clf_pred])

if __name__ == '__main__':
    # Test the DummyModel
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, (10, 2))
    model = DummyModel()
    model.fit(X, y)
    print(model.predict(X))
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.base import BaseEstimator

class GradientBoostingDual(BaseEstimator):
    def __init__(self, regressor_params=None, classifier_params=None):
        self.regressor = GradientBoostingRegressor(**(regressor_params or {}))
        self.classifier = GradientBoostingClassifier(**(classifier_params or {}))
    
    def fit(self, X, y_regressor, y_classifier):
        self.regressor.fit(X, y_regressor)
        self.classifier.fit(X, y_classifier)
        return self
    
    def predict(self, X):
        return self.regressor.predict(X), self.classifier.predict(X)
    

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    # Create some synthetic data for regression and classification
    X = np.random.rand(100, 5)
    y_regressor = np.random.rand(100)
    y_classifier = np.random.randint(0, 2, size=100)

    # Split data into training and test sets
    X_train, X_test, y_train_regressor, y_test_regressor, y_train_classifier, y_test_classifier = train_test_split(
        X, y_regressor, y_classifier, test_size=0.2, random_state=42
    )

    # Instantiate and train the dual model
    model = GradientBoostingDual()
    model.fit(X_train, y_train_regressor, y_train_classifier)

    # Predict with the model
    regressor_pred, classifier_pred = model.predict(X_test)

    # Output results
    print("Regressor predictions:", regressor_pred[:5])
    print("Classifier predictions:", classifier_pred[:5])
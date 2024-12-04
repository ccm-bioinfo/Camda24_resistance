from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from .dual import DualTaskMixin

class GradientBoostingDualModel(DualTaskMixin):
    def __init__(self, regressor_params=None, classifier_params=None, random_state=42):
        super().__init__(regressor_params, classifier_params, random_state, 
                        regressor_estimator=GradientBoostingRegressor, classifier_estimator=GradientBoostingClassifier)
    
    

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
    model = GradientBoostingDualModel()
    model.fit(X_train, y_train_regressor, y_train_classifier)

    # Predict with the model
    regressor_pred, classifier_pred = model.predict(X_test)

    # Output results
    print("Regressor predictions:", regressor_pred[:5])
    print("Classifier predictions:", classifier_pred[:5])
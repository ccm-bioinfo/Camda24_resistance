from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_random_state

class DualTaskMixin(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, mic_hyperparams=None, phenotype_hyperparams=None, random_state=None, 
            mic_estimator=None, phenotype_estimator=None):
        self.mic_hyperparams = mic_hyperparams
        self.phenotype_hyperparams = phenotype_hyperparams
        self.random_state = random_state
        self.mic_estimator = mic_estimator
        self.phenotype_estimator = phenotype_estimator
        self.random_state = random_state

    def fit(self, X, y):
        # Set random_state_ using check_random_state
        self.random_state_generator = check_random_state(self.random_state)
        # Add random_state to hyperparameters for each estimator
        mic_params = self.add_random_state_if_missing(self.mic_hyperparams)
        phenotype_params = self.add_random_state_if_missing(self.phenotype_hyperparams)
        # Initialize estimators with the hyperparameters and random state
        self.mic_estimator_instance = self.mic_estimator(**mic_params)
        self.phenotype_estimator_instance = self.phenotype_estimator(**phenotype_params)
        
        # Fit the models
        self.mic_estimator_instance.fit(X, y['mic_labels_encoded'])
        self.phenotype_estimator_instance.fit(X, y['phenotype_labels_encoded'])
        return self

    def add_random_state_if_missing(self, params):
        if params is None:
            params = {}
        if 'random_state' not in params:
            # Ensure the random_state is an integer for consistency across tasks
            params['random_state'] = self.random_state_generator.randint(0, 2**32 - 1)
        return params

    def predict(self, X):
        mic_predictions = self.mic_estimator_instance.predict(X)
        phenotype_predictions = self.phenotype_estimator_instance.predict(X)
        return mic_predictions, phenotype_predictions

    def predict_proba(self, X):
        if hasattr(self.mic_estimator_instance, 'predict_proba'):
            mic_proba = self.mic_estimator_instance.predict_proba(X)
        else:
            print("The regressor does not have a predict_proba method.")
            mic_proba = None
        if hasattr(self.phenotype_estimator_instance, 'predict_proba'):
            phenotype_proba = self.phenotype_estimator_instance.predict_proba(X)
        else:
            print("The classifier does not have a predict_proba method.")
            phenotype_proba = None
        return mic_proba, phenotype_proba  # Return None for regressor's predict_proba

    def score(self, X, y):
        mic_score = self.mic_estimator_instance.score(X, y['mic_labels_encoded'])
        phenotype_score = self.phenotype_estimator_instance.score(X, y['phenotype_labels_encoded'])
        return mic_score, phenotype_score

# Example usage:
if __name__ == '__main__':
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    import pandas as pd
    import numpy as np

    # Sample data
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.DataFrame({
        'mic_labels_encoded': np.random.rand(100),
        'phenotype_labels_encoded': np.random.randint(0, 2, 100)  # Binary classification
    })

    # Initialize the DualTaskMixin with RandomForest estimators for regression and classification
    dual_task_model = DualTaskMixin(
        mic_estimator=RandomForestRegressor,
        phenotype_estimator=RandomForestClassifier,
        mic_hyperparams={'n_estimators': 100},
        phenotype_hyperparams={'n_estimators': 100},
        random_state=42
    )

    # Fit the model
    dual_task_model.fit(X, y)

    # Predictions
    mic_predictions, phenotype_predictions = dual_task_model.predict(X)
    print(f'MIC Predictions: {mic_predictions[:5]}')
    print(f'Phenotype Predictions: {phenotype_predictions[:5]}')

    # Score
    mic_score, phenotype_score = dual_task_model.score(X, y)
    print(f'MIC Score: {mic_score}')
    print(f'Phenotype Score: {phenotype_score}')
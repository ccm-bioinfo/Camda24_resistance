from .meta_estimator.dual import DualTaskMixin
from sklearn.dummy import DummyRegressor, DummyClassifier

class DummyDualModel(DualTaskMixin):
    def __init__(self, mic_hyperparams=None, phenotype_hyperparams=None, random_state=42):
        super().__init__(mic_hyperparams, phenotype_hyperparams, random_state, 
                        mic_estimator=DummyRegressor, phenotype_estimator=DummyClassifier)
        
    def fit(self, X, y):
        # Set mic estimator
        mic_estimator_type = self.mic_hyperparams.get('estimator', 'regressor')
        self.mic_estimator = DummyRegressor if mic_estimator_type == 'regressor' else DummyClassifier
        # Set phenotype estimator
        phenotype_estimator_type = self.phenotype_hyperparams.get('estimator', 'classifier')
        self.phenotype_estimator = DummyClassifier if phenotype_estimator_type == 'classifier' else DummyRegressor
        return super().fit(X, y)

if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Create synthetic data for regression and classification
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, (100, 2))

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train the model
    model = DummyDualModel(mic_hyperparams={'strategy': 'mean'}, phenotype_hyperparams={'strategy': 'most_frequent'})
    model.fit(X_train, y_train)

    # Predict with the trained model
    regression_predictions, classification_predictions = model.predict(X_test)

    # Output the results
    print("Regression predictions:", regression_predictions[:5])
    print("Classification predictions:", classification_predictions[:5])


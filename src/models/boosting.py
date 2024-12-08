from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from .dual import DualTaskMixin

class GradientBoostingDualModel(DualTaskMixin):
    def __init__(self, mic_hyperparams=None, phenotype_hyperparams=None, random_state=42):
        super().__init__(mic_hyperparams, phenotype_hyperparams, random_state, 
                        mic_estimator=GradientBoostingRegressor, phenotype_estimator=GradientBoostingClassifier)
    
    

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    # Create some synthetic data for regression and classification
    X = np.random.rand(100, 5)
    mic = np.random.rand(100)
    phenotype = np.random.randint(0, 2, size=100)

    # Split data into training and test sets
    X_train, X_test, mic_train, mic_test, phenotype_train,phenotype_test = train_test_split(
        X, mic, phenotype, test_size=0.2, random_state=42
    )

    # Instantiate and train the dual model
    model = GradientBoostingDualModel()
    model.fit(X_train, mic_train, phenotype_train)

    # Predict with the model
    mic_pred, phenotype_pred = model.predict(X_test)

    # Output results
    print("Regressor predictions:", mic_pred[:5])
    print("Classifier predictions:", phenotype_pred[:5])
from sklearn.linear_model import LinearRegression, LogisticRegression
from .dual import DualTaskMixin

class LinearDualModel(DualTaskMixin):
    def __init__(self, mic_hyperparams=None, phenotype_hyperparams=None, random_state=42):
        phenotype_hyperparams['penalty'] = 'none' if phenotype_hyperparams is None else phenotype_hyperparams.get('penalty', 'none')
        super().__init__(mic_hyperparams, phenotype_hyperparams, random_state, 
                        mic_estimator=LinearRegression, phenotype_estimator=LogisticRegression)

if __name__ == '__main__':
    # Test the LinearModel
    import numpy as np
    X = np.random.rand(100, 5)
    y_mic = np.random.rand(100, 1)
    y_phenotype = np.random.randint(0, 2, (100, 1))
    y = (y_mic, y_phenotype)
    model = LinearDualModel()
    model.fit(X, y)
    print(model.predict(X))
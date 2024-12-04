from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from .dual import DualTaskMixin

class KnnDualModel(DualTaskMixin):
    def __init__(self, regressor_params=None, classifier_params=None, random_state=42):
        super().__init__(regressor_params, classifier_params, random_state, 
                        regressor_estimator=KNeighborsRegressor, classifier_estimator=KNeighborsClassifier)

if __name__ == '__main__':
    import numpy as np
    # Test the KNNModel
    X = np.random.rand(10, 5)
    y = np.hstack([
        np.random.rand(10, 1),
        np.random.randint(0, 2, (10, 1))
    ])
    model = KnnDualModel(regressor_params={'n_neighbors': 3}, classifier_params={'n_neighbors': 2})
    model.fit(X, y)
    print(model.predict(X))
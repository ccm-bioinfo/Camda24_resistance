import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class SupportVectorModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, kernel='linear', C=1.0):
        """
        Initializes Support Vector models for regression and classification.

        Args:
            kernel (str): Kernel type to be used in SVR and SVC.
            C (float): Regularization parameter for SVR and SVC.
        """
        self.kernel = kernel
        self.C = C
        self.svr = SVR(kernel=kernel, C=C)
        self.svc = SVC(kernel=kernel, C=C)

    def fit(self, X, y):
        """
        Fits the model with the provided data.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target matrix where the last column is for classification.
        
        Returns:
            self: Fitted model.
        """
        y_reg = y[:, :-1]  # All columns except the last for regression
        y_clf = y[:, -1]   # Last column for classification (binary or multiclass)

        # Fit models
        self.svr.fit(X, y_reg.ravel())  # SVR expects 1D target
        self.svc.fit(X, y_clf)
        
        return self

    def predict(self, X):
        """
        Predicts regression and classification outputs for the given data.

        Args:
            X (ndarray): Feature matrix.
        
        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        reg_pred = self.svr.predict(X).reshape(-1, 1)
        clf_pred = self.svc.predict(X).reshape(-1, 1)
        return np.column_stack([reg_pred, clf_pred])

if __name__ == '__main__':
    # Test the SupportVectorModel
    X = np.random.rand(10, 5)
    y = np.hstack([
        np.random.rand(10, 1),    # Regression target
        np.random.randint(0, 2, (10, 1))  # Classification target
    ])
    model = SupportVectorModel(kernel='rbf', C=1.0)
    model.fit(X, y)
    print(model.predict(X))
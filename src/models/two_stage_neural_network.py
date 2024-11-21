import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class TwoStageNeuralNetworkModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, input_dim, hidden_units=64, learning_rate=0.001, epochs=20, batch_size=32):
        """
        Initializes two-stage Neural Network models for regression and classification.

        Args:
            input_dim (int): Number of input features.
            hidden_units (int): Number of units in hidden layers.
            learning_rate (float): Learning rate for training.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # First-stage models
        self.dominant_classifier = self._build_classifier()
        self.dominant_regressor = self._build_regressor()

        # Second-stage models
        self.refine_classifier = self._build_classifier()
        self.refine_regressor = self._build_regressor()

    def _build_classifier(self):
        """Builds a simple feedforward neural network for classification."""
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(self.hidden_units, activation='relu'),
            Dense(self.hidden_units, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_regressor(self):
        """Builds a simple feedforward neural network for regression."""
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(self.hidden_units, activation='relu'),
            Dense(self.hidden_units, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def fit(self, X, y):
        """
        Fits the two-stage model.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target matrix where the last column is for classification.

        Returns:
            self: Fitted model.
        """
        y_reg = y[:, :-1].ravel()  # All columns except the last for regression
        y_clf = y[:, -1]           # Last column for classification (binary)

        # Determine dominant class and regression value
        dominant_class = np.bincount(y_clf.astype(int)).argmax()
        dominant_indices = (y_clf == dominant_class)
        dominant_reg_value = np.median(y_reg[dominant_indices])

        self.dominant_class = dominant_class
        self.dominant_reg_value = dominant_reg_value

        # Prepare data for training first-stage models
        y_dom_clf = dominant_indices.astype(int)
        y_dom_reg = np.full(len(X), dominant_reg_value)

        # First-stage training
        self.dominant_classifier.fit(X, y_dom_clf, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.dominant_regressor.fit(X, y_dom_reg, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Prepare data for training second-stage models
        refine_indices = ~dominant_indices
        if np.any(refine_indices):
            self.refine_classifier.fit(X[refine_indices], y_clf[refine_indices], epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            self.refine_regressor.fit(X[refine_indices], y_reg[refine_indices], epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        
        return self

    def predict(self, X):
        """
        Predicts using the two-stage approach.

        Args:
            X (ndarray): Feature matrix.

        Returns:
            ndarray: Concatenated regression and classification predictions.
        """
        # Stage 1: Predict dominant class and value
        dominant_class_pred = (self.dominant_classifier.predict(X) > 0.5).astype(int).ravel()
        dominant_reg_pred = self.dominant_regressor.predict(X).ravel()

        # Stage 2: Refine for non-dominant cases
        non_dominant_indices = (dominant_class_pred == 0)
        refined_class_pred = dominant_class_pred.copy()
        refined_reg_pred = dominant_reg_pred.copy()

        if np.any(non_dominant_indices):
            refined_class_pred[non_dominant_indices] = (self.refine_classifier.predict(X[non_dominant_indices]) > 0.5).astype(int).ravel()
            refined_reg_pred[non_dominant_indices] = self.refine_regressor.predict(X[non_dominant_indices]).ravel()

        return np.column_stack([refined_reg_pred, refined_class_pred])

if __name__ == '__main__':
    # Test the TwoStageNeuralNetworkModel
    X = np.random.rand(100, 5)  # 5 features
    y = np.hstack([
        np.random.rand(100, 1),    # Regression target
        np.random.randint(0, 2, (100, 1))  # Classification target (binary)
    ])
    model = TwoStageNeuralNetworkModel(input_dim=5, hidden_units=64, epochs=10, batch_size=16)
    model.fit(X, y)
    print(model.predict(X))
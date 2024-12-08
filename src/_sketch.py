from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pandas as pd

# Custom Target Transformer
class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, apply_log=False):
        self.apply_log = apply_log

    def fit(self, y):
        return self

    def transform(self, y):
        y_reg, y_class = y
        if self.apply_log:
            y_reg = np.log1p(y_reg)
        return y_reg, y_class

    def inverse_transform(self, y):
        y_reg, y_class = y
        if self.apply_log:
            y_reg = np.expm1(y_reg)
        return y_reg, y_class

# Combined Estimator
class CombinedEstimator(BaseEstimator):
    def __init__(self, regressor=None, classifier=None, target_transformer=None):
        self.regressor = regressor
        self.classifier = classifier
        self.target_transformer = target_transformer

    def fit(self, X, y):
        y_reg, y_class = y
        self.regressor.fit(X, y_reg)
        self.classifier.fit(X, y_class)
        return self

    def predict(self, X):
        reg_pred = self.regressor.predict(X)
        class_pred = self.classifier.predict(X)
        if self.target_transformer:
            reg_pred, class_pred = self.target_transformer.inverse_transform((reg_pred, class_pred))
        return reg_pred, class_pred

# Custom scoring function for combined regressor and classifier
def custom_score(estimator, X, y):
    y_reg, y_class = y
    reg_pred, class_pred = estimator.predict(X)
    reg_score = -mean_squared_error(y_reg, reg_pred)
    class_score = accuracy_score(y_class, class_pred)
    return 0.5 * reg_score + 0.5 * class_score

# Example Data
X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [2, 4, 6, 8]})
y_reg = np.array([1, 2, 3, 4])  # Regression target
y_class = np.array([0, 1, 0, 1])  # Classification target
y = (y_reg, y_class)

# Instantiate components
regressor = RandomForestRegressor()
classifier = RandomForestClassifier()
target_transformer = TargetTransformer(apply_log=True)

# Separate pipeline for target transformer
target_transformer_pipeline = Pipeline([
    ('target_transformer', target_transformer)
])

# Full pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('target_transformer', target_transformer_pipeline),  # Target transformation
    ('model', CombinedEstimator(regressor=regressor, classifier=classifier, target_transformer=target_transformer))  # Combined model
])

# Define hyperparameter grid for both model and transformer
param_grid = {
    'target_transformer__target_transformer__apply_log': [True, False],  # Log transformation for targets
    'model__regressor__n_estimators': [50, 100],  # Number of trees for regressor
    'model__classifier__n_estimators': [50, 100],  # Number of trees for classifier
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring=custom_score, verbose=1)

# Fit the grid search
grid_search.fit(X, y)

# Best model after tuning
best_model = grid_search.best_estimator_

# Predictions with best model
y_pred = best_model.predict(X)

print("Best Parameters:", grid_search.best_params_)
print("Predictions (after inverse transform):")
print("Regression Predictions:", y_pred[0])
print("Classification Predictions:", y_pred[1])
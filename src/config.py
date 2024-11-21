from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import multiprocessing

# Base regressors
BASE_REGRESSORS = {
    'rf': RandomForestRegressor,
    'svm': SVR,
    'gb': GradientBoostingRegressor,
    'knn': KNeighborsRegressor
}

# Base classifiers
BASE_CLASSIFIERS = {
    'rf': RandomForestClassifier,
    'svm': SVC,
    'gb': GradientBoostingClassifier,
    'knn': KNeighborsClassifier
}

# Meta estimators
META_REGRESSOR = Ridge
META_CLASSIFIER = LogisticRegression

# Hyperparameter search space
SEARCH_BOUNDS = {
    'rf_n_estimators_reg': (50, 200),
    'svm_C': (0.001, 10.0),
    'gb_n_estimators_reg': (50, 200),
    'knn_n_neighbors_reg': (3, 10),
    'rf_n_estimators_clf': (50, 200),
    'gb_n_estimators_clf': (50, 200),
    'knn_n_neighbors_clf': (3, 10),
    'reg_alpha': (0.01, 10.0),
    'clf_C': (0.001, 10.0),
    'bagging_n_estimators': (5, 20)
}

DEVICE_CORES = multiprocessing.cpu_count()
MAX_CORES = max(1, DEVICE_CORES - 4)
DISCRETE_LIMIT = 15
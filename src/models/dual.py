from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class DualTaskMixin(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, mic_hyperparams=None, phenotype_hyperparams=None, random_state=42, 
                mic_estimator=None, phenotype_estimator=None):
        self.mic_hyperparams = self._add_random_state(mic_hyperparams, random_state)
        self.phenotype_hyperparams = self._add_random_state(phenotype_hyperparams, random_state)
        self.mic_estimator = mic_estimator(self.mic_hyperparams)
        self.phenotype_estimator = phenotype_estimator(self.phenotype_hyperparams)

    def _add_random_state(self, params, random_state):
        if params is None:
            params = {}
        if 'random_state' not in params:
            params['random_state'] = random_state
        return params

    def fit(self, X, y):
        self.regressor.fit(X, y['mic_labels_encoded'])
        self.classifier.fit(X, y['phenotype_labels_encoded'])
        return self

    def predict(self, X):
        return self.regressor.predict(X), self.classifier.predict(X)
    
    def predict_proba(self, X):
        return self.regressor.predict_proba(X), self.classifier.predict_proba(X)
    
    def score(self, X, y):
        return self.regressor.score(X, y['mic_labels_encoded']), self.classifier.score(X, y['phenotype_labels_encoded'])
    
    if __name__ == '__main__':
        pass
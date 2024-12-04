from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .tune import Tuner

class NestedCrossValidationTuner:
    def __init__(self, model, param_space, validation_fn, outer_splits=5, inner_splits=3, n_trials=10):
        self.model = model
        self.param_space = param_space
        self.validation_fn = validation_fn
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.n_trials = n_trials

    def tune(self, X, y):
        outer_skf = StratifiedKFold(n_splits=self.outer_splits, shuffle=True, random_state=42)
        outer_scores = []

        for train_index, test_index in outer_skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            tuner = Tuner(
                self.model,
                lambda m, Xt, yt, Xv, yv: self.validation_fn(m, Xt, yt, n_splits=self.inner_splits),
                self.param_space
            )
            best_params, _ = tuner.tune(X_train, y_train, None, None, n_trials=self.n_trials)
            
            self.model.set_params(**best_params)
            self.model.fit(X_train, y_train)
            
            test_score = accuracy_score(y_test, self.model.predict(X_test))
            outer_scores.append(test_score)
        
        return outer_scores

if __name__ == '__main__':
    pass
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.datasets import load_iris

    # X, y = load_iris(return_X_y=True)
    # model = LogisticRegression()
    # param_space = lambda trial: {
    #     'C': trial.suggest_loguniform('C', 1e-10, 1e10),
    #     'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
    # }
    # validation_fn = lambda m, X, y, n_splits: accuracy_score(y, m.predict(X))
    # ncv_tuner = NestedCrossValidationTuner(model, param_space, validation_fn)
    # scores = ncv_tuner.tune(X, y)
    # print(scores)
import optuna
import mlflow

from .evaluate import PredictionEvaluator
from .validation import CrossValidation
class Tuner:
    def __init__(self, model, x, y, validation_function, param_space):
        self.model = model
        self.validation_function = validation_function
        self.param_space = param_space
        self.x = x
        self.y = y

    def objective(self, trial):
        params = self.param_space(trial)
        self.model.set_params(**params)
        self.model.fit(X_train, y_train)
        cv = CrossValidation(self.model, self.x, self.y)
        metrics_df = cv[self.validation_function](self.model,)
        return metrics_df

    def tune(self, X_train, y_train, n_trials=20):
        with mlflow.start_run() as run:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train),
                n_trials=n_trials
            )
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_validation_score", study.best_value)
            return study.best_params, study.best_value
        
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
    # tuner = Tune(model, validation_fn, param_space)
    # best_params, best_score = tuner.tune(X, y, None, None, n_trials=10)
    # print(best_params, best_score)
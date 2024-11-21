import numpy as np
import optuna
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import f1_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, StratifiedKFold
from models.stacking_dev_model import MultiOutputModel
from evaluate import evaluate_predictions
from config import BASE_REGRESSORS, BASE_CLASSIFIERS, META_REGRESSOR, META_CLASSIFIER, SEARCH_BOUNDS, MAX_CORES
import json
import joblib  # For saving models

class Tuner:
    def __init__(self, X, y_class, y_reg, n_trials=10, n_splits=5, scorer_type="classification"):
        self.X = X
        self.y_class = y_class
        self.y_reg = y_reg
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.scorer_type = scorer_type
        self.study = None
        self.best_params = None
        self.best_model = None
        self.best_score = -np.inf
        self.best_metrics = None
        self.score_name = 'f1' if self.scorer_type == "classification" else 'MAPE'
        self.comparator = 1 if self.scorer_type == "classification" else -1

    def create_trial_model(self, trial):
        """
        Create the MultiOutputModel with parameters suggested by Optuna for the current trial.
        """
        trial_candidates = {param: trial.suggest_float(param, *range) if isinstance(range, tuple) else trial.suggest_int(param, *range)
                            for param, range in SEARCH_BOUNDS.items()}
        
        # Instantiate base and meta estimators using trial_candidates
        base_estimators_reg = [(name, model(n_estimators=trial_candidates[f'{name}_n_estimators_reg']))
                               for name, model in BASE_REGRESSORS.items()]
        base_estimators_clf = [(name, model(n_estimators=trial_candidates[f'{name}_n_estimators_clf'], class_weight='balanced'))
                               for name, model in BASE_CLASSIFIERS.items()]
        
        meta_regressor = BaggingRegressor(META_REGRESSOR(alpha=trial_candidates['reg_alpha']),
                                          n_estimators=trial_candidates['bagging_n_estimators'])
        meta_classifier = BaggingClassifier(META_CLASSIFIER(C=trial_candidates['clf_C']),
                                            n_estimators=trial_candidates['bagging_n_estimators'])
        
        return MultiOutputModel(
            base_estimators_reg=base_estimators_reg,
            base_estimators_clf=base_estimators_clf,
            meta_regressor=meta_regressor,
            meta_classifier=meta_classifier,
            use_oversampling=True,
            use_stacking=True
        )

    def objective(self, trial, X_train, y_train_combined):
        """
        Objective function for hyperparameter optimization with inner cross-validation.
        """
        model = self.create_trial_model(trial)
        inner_cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        classification_scores = []
        regression_scores = []
        for train_idx, val_idx in inner_cv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold_combined, _ = y_train_combined[train_idx], y_train_combined[val_idx]
            model.fit(X_train_fold, y_train_fold_combined)
            y_pred_combined = model.predict(X_val_fold)
            y_reg_pred, y_class_pred = y_pred_combined[:, 0], y_pred_combined[:, 1:]
            y_class_pred_labels = np.argmax(y_class_pred, axis=1)
            classification_score = f1_score(self.y_class[val_idx], y_class_pred_labels, average='macro')
            regression_scores = mean_absolute_percentage_error(self.y_reg[val_idx], y_reg_pred)
            classification_scores.append(classification_score)
            regression_scores.append(regression_scores)
        return np.mean(classification_scores), np.mean(regression_scores)

    def perform_nested_cv(self):
        """
        Perform nested cross-validation with Optuna hyperparameter tuning.
        """
        outer_cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in outer_cv.split(self.X, self.y_class):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_class_train, y_class_test = self.y_class[train_idx], self.y_class[test_idx]
            y_reg_train, y_reg_test = self.y_reg[train_idx], self.y_reg[test_idx]
            # Optimize hyperparameters using Optuna on the training data of the outer fold
            self.study = optuna.create_study(direction=["maximize", "maximize"], study_name="resistance_hyperparameter_optimization")
            self.study.optimize(lambda trial: self.objective(trial, X_train, np.column_stack([y_reg_train, y_class_train])), 
                                n_trials=self.n_trials, n_jobs=MAX_CORES, timeout=4000)      
            current_tuned_params = self.study.best_params
            # Create model with current tuned params found
            current_tuned_model = self.create_trial_model(optuna.trial.FixedTrial(current_tuned_params))
            current_tuned_model.fit(X_train, np.column_stack([y_reg_train, y_class_train]))
            current_preds = current_tuned_model.predict(X_test)
            current_metrics = evaluate_predictions(current_preds, y_reg_test, y_class_test)
            current_score = current_metrics[self.score_name]
            if current_score * self.comparator > self.best_score:
                self.best_score = current_score
                self.best_params = current_tuned_params
                self.best_model = current_tuned_model
                self.best_metrics = current_metrics

    def save(self, file_model='best_model.pkl', file_params='best_params.json'):
        """
        Save the best model and its parameters to files.
        """
        if self.best_model is not None and self.best_params is not None:
            # Save the best model to a file
            joblib.dump(self.best_model, file_model)
            # Save the best hyperparameters to a JSON file
            with open(file_params, 'w') as f:
                json.dump(self.best_params, f)

    def tune(self):
        """
        Perform the full hyperparameter tuning process with nested cross-validation.
        """
        self.perform_nested_cv()
        self.save()  # Save the best model and params after the tuning process
        return self.best_metrics, self.best_params
    
if __name__ == "__main__":

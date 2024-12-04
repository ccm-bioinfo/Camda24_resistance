from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, StratifiedKFold, RepeatedKFold, GroupKFold
from sklearn.model_selection import train_test_split

from .evaluate import PredictionEvaluator

class CrossValidation:
    def __init__(self, model, X, y, random_state=42, n_splits=5, leave_out_p=10, mic_n_classes = None, test_size=0.2, groups=None):
        self.model = model
        self.X = X
        self.y = y
        self.random_state = random_state
        self.mic_n_classes = mic_n_classes
        self.n_splits = n_splits
        self.leave_out_p = leave_out_p
        self.test_size = test_size
        self.groups = groups
        self.prediction_evaluator = PredictionEvaluator(self.mic_n_classes)

    def simple(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        metrics = self.prediction_evaluator.evaluate(self.model.predict_proba(X_test), y_test)
        metrics['model_score'] = self.model.score(X_test, y_test)
        return metrics

    def k_fold(self):
        kf = KFold(n_splits=self.inner_n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train)
            metrics = self.prediction_evaluator.evaluate(self.model.predict_proba(X_test), y_test)
            metrics['model_score'] = self.model.score(X_test, y_test)
            scores.append(metrics)
        return scores

    def loo(self):
        loo = LeaveOneOut()
        scores = []
        for train_index, test_index in loo.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train)
            metrics = self.prediction_evaluator.evaluate(self.model.predict_proba(X_test), y_test)
            metrics['model_score'] = self.model.score(X_test, y_test)
            scores.append(metrics)
        return scores

    def lpo(self):
        lpo = LeavePOut(p=self.leave_out_p)
        scores = []
        for train_index, test_index in lpo.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train)
            metrics = self.prediction_evaluator.evaluate(self.model.predict_proba(X_test), y_test)
            metrics['model_score'] = self.model.score(X_test, y_test)
            scores.append(metrics)
        return scores

    def stratified_k_fold(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train)
            metrics = self.prediction_evaluator.evaluate(self.model.predict_proba(X_test), y_test)
            metrics['model_score'] = self.model.score(X_test, y_test)
            scores.append(metrics)
        return scores

    def repeated_k_fold(self):
        rkf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        scores = []
        for train_index, test_index in rkf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train)
            metrics = self.prediction_evaluator.evaluate(self.model.predict_proba(X_test), y_test)
            metrics['model_score'] = self.model.score(X_test, y_test)
            scores.append(metrics)
        return scores

    def group_k_fold(self):
        gkf = GroupKFold(n_splits=self.n_splits)
        scores = []
        for train_index, test_index in gkf.split(self.X, self.y, groups=self.groups):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train)
            metrics = self.prediction_evaluator.evaluate(self.model.predict_proba(X_test), y_test)
            metrics['model_score'] = self.model.score(X_test, y_test)
            scores.append(metrics)
        return scores

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    X, y = load_iris(return_X_y=True)
    
    # Initialize model and cross-validation class
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = CrossValidation(model, X, y)
    
    # Run different cross-validation methods
    print("Simple CV:", cv.simple())
    print("K-Fold CV:", cv.k_fold())
    print("LOO CV:", cv.loo())
    print("LPO CV:", cv.lpo())
    print("Nested CV:", cv.nested())
    print("Stratified K-Fold CV:", cv.stratified_k_fold())
    print("Repeated K-Fold CV:", cv.repeated_k_fold())
    print("Group K-Fold CV:", cv.group_k_fold(groups=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
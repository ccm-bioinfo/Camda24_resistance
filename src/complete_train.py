from preprocess.feature import FeaturesTransformer
from preprocess.target import TargetTransformer
from .oversample import Oversampler

class CompleteTrainer:
    def __init__(self, model, df, oversample_target_index = 1):
        self.df = df
        self.model = model
        self.oversample_target_index = oversample_target_index
        self.x_transformer = FeaturesTransformer()
        self.y_transformer= TargetTransformer()

    def _extract_xy(self):
        self.df = self.df.dropna()
        x = self.df.drop(columns=['mic', 'phenotype'])
        if 'mic' in self.df and 'phenotype' in self.df:
            y = self.df[['mic', 'phenotype']]
        else:
            y = None
        return x, y

    def _oversample(self):
        ros = Oversampler()
        return ros.fit_resample(self.x_transformed, self.y_transformed[self.oversample_target_index])

    def _preprocess(self, x, y = None):
        x_transformed = self.x_transformer.fit_transform(x)
        y_transformed = self.y_transformer.fit_transform(y) if y is not None else None
        return x_transformed, y_transformed

    def _get_training_data(self, x_transformed, y_transformed):
        return self._oversample() if self.oversample_target_index else (x_transformed, y_transformed)

    def fit(self):
        x, y = self._extract_xy(self.df)
        x_transformed, y_transformed = self._preprocess(x, y)
        x_train, y_train= self._get_training_data(x_transformed, y_transformed)
        self.model.fit(x_train, y_train)
        return self

    def train(self):
        return self.fit()

    def predict(self, x):
        x_transformed, _ = self.preprocess(x, None)
        y_transformed_estimate = self.model.predict(x_transformed)
        y_estimate = self.target_preprocessor.inverse_transform(y_transformed_estimate)
        return y_estimate

    def predict_proba(self, x):
        x_transformed, _ = self.preprocess(x, None)
        y_transformed_estimate = self.model.predict_proba(x_transformed)
        y_estimate = self.target_preprocessor.inverse_transform(y_transformed_estimate)
        return y_estimate
    

if __name__ == '__main__':
    from models.dummy import DummyDualModel
    import numpy as np
    import pandas as pd

    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, (10, 2))
    df = pd.DataFrame(np.column_stack([X, y]), columns=['f1', 'f2', 'f3', 'f4', 'f5', 'mic', 'phenotype'])
    model = DummyDualModel(mic_hyperparams={'strategy' : 'median'}, phenotype_hyperparams={'strategy' : 'most_frequent'})
    trainer = CompleteTrainer(model, df)
    trainer.train()
    print(trainer)

# main.py

import pandas as pd
from prepare import create_preprocessor_X, create_preprocessor_y
from models.stacking_dev_model import MultiOutputModel, ClusterAugmentEncoder
from tune import Tuner

class ModelSearcher:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y_class = None
        self.y_reg = None
        self.X_transformed = None
        self.phenotype_transformed = None
        self.mic_transformed = None
        self.best_hyperparameters = None
        self.all_metrics = None

    def load_data(self):
        """Load and clean the dataset."""
        self.df = pd.read_csv(self.data_path, sep='\t', low_memory=False)
        self.df = self.df.dropna()

    def preprocess_data(self):
        """Separate features and target, and preprocess them."""
        self.X = self.df.drop(columns=['mic', 'phenotype', 'accession'])
        self.y_class = self.df[['phenotype']]
        self.y_reg = self.df[['mic']]

        preprocessor_X = create_preprocessor_X(self.df)
        phenotype_pipeline, mic_pipeline = create_preprocessor_y()

        self.X_transformed = preprocessor_X.fit_transform(self.X)
        self.phenotype_transformed = phenotype_pipeline.fit_transform(self.df[['phenotype']])
        self.mic_transformed = mic_pipeline.fit_transform(self.df[['mic']])

    def optimize_hyperparameters(self):
        """Run Optuna for hyperparameter optimization."""
        self.y_class = self.phenotype_transformed.to_numpy().flatten() if isinstance(self.phenotype_transformed, pd.DataFrame) else self.phenotype_transformed
        self.y_reg = self.mic_transformed.to_numpy().flatten() if isinstance(self.mic_transformed, pd.DataFrame) else self.mic_transformed
        
        self.all_metrics, self.best_hyperparameters = Tuner.tune(self.X_transformed, self.y_class, self.y_reg, n_trials=10, n_splits=5)
        
    def save_results(self):
        """Save metrics and best hyperparameters."""
        print("Best Hyperparameters:", self.best_hyperparameters)
        self.all_metrics.to_csv("./all_metrics.csv", index=False)

    def run_search(self):
        """Execute the full hyperparameter search process."""
        self.load_data()
        self.preprocess_data()
        self.optimize_hyperparameters()
        self.save_results()


if __name__ == "__main__":
    search = ModelSearcher('./../../../DataSets/group-2/data/combined_antibiotic_resistance.tsv')
    search.run_search()

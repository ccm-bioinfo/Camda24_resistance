import os
import pandas as pd
from .models import Model1, Model2  # Example models from models.py
from .tuner import Tuner

class Experimenter:
    def __init__(self, data_path: str, models_path: str, validation_type='standard', evaluation_metrics=None, n_trials=50):
        self.data_path = data_path
        self.models_path = models_path
        self.validation_type = validation_type
        self.evaluation_metrics = evaluation_metrics or ['accuracy']
        self.n_trials = n_trials
        self.df = self._load_data()

    def _load_data(self):
        """
        Load and prepare data, optionally split by antibiotic if needed.
        """
        df = pd.read_csv(self.data_path)
        
        # Example of splitting by antibiotic (if needed)
        if 'antibiotic' in df.columns:
            self.df_no_antibiotic = df[df['antibiotic'] == 'None']
            self.df_antibiotic = df[df['antibiotic'] != 'None']
        else:
            self.df_no_antibiotic = df
            self.df_antibiotic = None
        
        return df

    def _load_models(self):
        """
        Dynamically load models from models_path (e.g., Model1, Model2 from models.py)
        """
        # For simplicity, assuming the models are imported directly in models.py
        models = [Model1(), Model2()]  # This is a placeholder for actual model classes
        return models

    def execute(self):
        """
        Loop through models, initialize the Tuner, and execute the tuning process.
        """
        results = []

        # Load models
        models = self._load_models()

        for model in models:
            print(f"Starting tuning for {model.__class__.__name__}")
            
            # Initialize the tuner with the current model
            tuner = Tuner(model=model,
                df=self.df,  # Use whole dataset or split as needed
                validation_type=self.validation_type,
                n_trials=self.n_trials,
                evaluation_metrics=self.evaluation_metrics)
            
            # Execute the tuner and retrieve the results
            best_params, best_values = tuner.tune()

            # Store results in the results list
            results.append({
                'model': model.__class__.__name__,
                'best_params': best_params,
                'best_values': best_values
            })
        
        # Save the results to a CSV file
        self._save_results(results)

    def _save_results(self, results):
        """
        Save results to a CSV file.
        """
        result_df = pd.DataFrame(results)
        result_path = os.path.join(self.models_path, "tuning_results.csv")
        result_df.to_csv(result_path, index=False)
        print(f"Results saved to {result_path}")

# Experimenter will loop over models
# 1. Load the data (split or not by antibiotic)
# 2. loop over model in models.py
# 3. Call Tuner
# 4. Save the results

# experimenter = Experimenter(data_path, models_path)
# experimenter.execute()
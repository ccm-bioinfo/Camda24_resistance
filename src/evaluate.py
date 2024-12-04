import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, log_loss, average_precision_score, r2_score
)

class PredictionEvaluator:
    def __init__(self, mic_n_classes=None, quantile=0.5):
        self.mic_n_classes = mic_n_classes
        self.quantile = quantile

    @staticmethod
    def _regression_metrics(y_true, y_pred):
        wmae = np.average(np.abs(y_pred - y_true))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'WMAE': wmae,
            'MAPE': mape,
            'RMSE': rmse,
            'R2': r2
        }

    @staticmethod
    def _classification_metrics(y_true, y_pred, n_classes=2):
        is_soft_predictions = (
            np.all(y_pred >= 0) and 
            np.all(y_pred <= 1) and 
            (np.isclose(np.sum(y_pred, axis=1), 1).all() if y_pred.ndim > 1 else True)
        )
        
        metrics = {}
        
        if is_soft_predictions:
            if n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_pred[:, 1])
                y_pred_hard = (y_pred[:, 1] >= 0.5).astype(int)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr')
                metrics['average_precision'] = average_precision_score(y_true, y_pred, average='macro')
                y_pred_hard = np.argmax(y_pred, axis=1)
        else:
            y_pred_hard = y_pred
        
        metrics.update({
            'precision': precision_score(y_true, y_pred_hard),
            'recall': recall_score(y_true, y_pred_hard),
            'f1': f1_score(y_true, y_pred_hard),
            'accuracy': accuracy_score(y_true, y_pred_hard),
            'weighted_f1': f1_score(y_true, y_pred_hard, average='weighted')
        })
        return metrics

    @staticmethod
    def _quantile_loss(y_true, y_pred, quantile):
        error = y_true - y_pred
        loss = np.maximum(quantile * error, (quantile - 1) * error)
        return np.mean(loss)

    def evaluate(self, y_true, y_pred):
        y_mic_true, y_phenotype_true = y_true
        y_mic_pred, y_phenotype_pred = y_pred

        if self.mic_n_classes:
            mic_metrics = self._classification_metrics(y_mic_true, y_mic_pred, n_classes=self.mic_n_classes)
        else:
            mic_metrics = self._regression_metrics(y_mic_true, y_mic_pred)
            mic_metrics['quantile_loss'] = self._quantile_loss(y_mic_true, y_mic_pred, self.quantile)
        
        phenotype_metrics = self._classification_metrics(y_phenotype_true, y_phenotype_pred, n_classes=2)

        metrics = {**mic_metrics, **phenotype_metrics}
        return pd.DataFrame([metrics])


if __name__ == "__main__":
    # Example data
    y_mic = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_phenotype = np.array([0, 1, 0, 1, 0])
    y_true = (y_mic, y_phenotype)
    
    mic_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    phenotype_pred = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4]])
    y_pred = (mic_pred, phenotype_pred)
    
    # Initialize evaluator and evaluate predictions
    evaluator = PredictionEvaluator(mic_is_class=False, quantile=0.5)
    metrics_df = evaluator.evaluate(y_true, y_pred)
    print(metrics_df)
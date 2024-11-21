import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score


def evaluate_predictions(pred, y_reg, y_clf, quantile=0.5):
    """
    Evaluate regression and classification predictions using various metrics.

    Parameters:
    - pred: The model predictions, where the first column is regression predictions and the second is classification probabilities.
    - y_reg: Actual regression target values.
    - y_clf: Actual classification target values.
    - quantile: The quantile value for quantile loss (default is 0.5 for median).

    Returns:
    - metrics_df: DataFrame containing evaluation metrics.
    """
    # Separate regression and classification predictions
    reg_pred = pred[:, 0]  # Regression predictions
    clf_pred = pred[:, 1] > 0.5  # Binary classification predictions (threshold at 0.5)
    
    # Regression Metrics
    wmae = np.average(np.abs(reg_pred - y_reg))  # Weighted Mean Absolute Error (WMAE)
    mape = np.mean(np.abs((y_reg - reg_pred) / (y_reg + 1e-6))) * 100  # Mean Absolute Percentage Error (MAPE)
    rmse = np.sqrt(np.mean((y_reg - reg_pred)**2))  # Root Mean Squared Error (RMSE)
    r2 = r2_score(y_reg, reg_pred)  # R-squared (R2)
    
    # Classification Metrics
    precision = precision_score(y_clf, clf_pred)  # Precision
    recall = recall_score(y_clf, clf_pred)  # Recall
    f1 = f1_score(y_clf, clf_pred)  # F1 Score
    accuracy = accuracy_score(y_clf, clf_pred)  # Accuracy
    
    # Weighted Metrics
    weighted_f1 = f1_score(y_clf, clf_pred, average='weighted')  # Weighted F1 Score
    weighted_mape = np.average(mape, weights=np.ones_like(y_clf))  # Weighted MAPE

    def quantile_loss(y_true, y_pred, quantile):
        """
        Compute the quantile loss, also known as the pinball loss.

        Parameters:
        - y_true: The true values (e.g., regression target values).
        - y_pred: The predicted values (e.g., regression predictions).
        - quantile: The quantile to compute the loss for (default is 0.5 for median).

        Returns:
        - quantile_loss: The computed quantile loss value.
        """
        error = y_true - y_pred
        loss = np.maximum(quantile * error, (quantile - 1) * error)
        return np.mean(loss)

    quantile_loss_value = quantile_loss(y_reg, reg_pred, quantile)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'WMAE': wmae,
        'MAPE': mape,
        'RMSE': rmse,
        'R2': r2,
        'weighted_f1': weighted_f1,
        'weighted_mape': weighted_mape,
        'quantile_loss': quantile_loss_value
    }

    # Convert the metrics dictionary into a DataFrame and return
    return pd.DataFrame([metrics])

if __name__ == "__main__":
    # Example data
    y_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_clf = np.array([0, 1, 0, 1, 0])
    pred = np.array([
        [1.1, 0.9],
        [2.1, 0.8],
        [3.1, 0.2],
        [4.1, 0.7],
        [5.1, 0.1]
    ])

    # Evaluate the predictions
    metrics_df = evaluate_predictions(pred, y_reg, y_clf)
    print(metrics_df)
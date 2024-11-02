import pickle
import pandas as pd
import numpy as np
import random

from sklearn.metrics import (
    fbeta_score,
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from data_preprocessing import StockDataPreprocessor


class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model = pickle.load(open(model_path, "rb"))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates model performance using multiple metrics

        Returns:
            dict: Dictionary containing various performance metrics
        """
        y_pred = self.model.predict(X_test)
        y_true = y_test.values.astype(int)

        metrics = {
            "F2 Score": fbeta_score(y_true, y_pred, beta=2),
            "F1 Score": fbeta_score(y_true, y_pred, beta=1),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        }

        return metrics

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plots confusion matrix for model predictions
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        y_pred = self.model.predict(X_test)
        y_true = y_test.values.astype(int)

        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Event", "Extreme Event"],
            yticklabels=["No Event", "Extreme Event"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Initialize data preprocessor and load data
    preprocessor = StockDataPreprocessor(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-31"
    )
    stock_prices = preprocessor.download_and_prepare_stock_data()

    # Split data into train, validation and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(
        stock_prices,
        ["Open", "High", "Low", "Close", "Volume", "Daily_Returns"],
        "Extreme_Event",
        train_ratio=0.7,
        val_ratio=0.85,
    )

    # Convert time series data into supervised learning format
    X_train, y_train = preprocessor.time_series_to_supervised(
        X_train, y_train, lookback=10
    )
    X_val, y_val = preprocessor.time_series_to_supervised(X_val, y_val, lookback=10)
    X_test, y_test = preprocessor.time_series_to_supervised(X_test, y_test, lookback=10)

    # Initialize evaluator and compute metrics
    evaluator_rf = ModelEvaluator(model_path="models/best_random_forest.pkl")
    metrics = evaluator_rf.evaluate(X_test, y_test)

    # Print results
    print("\nModel Evaluation Results:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    print("-" * 40)

    # Plot confusion matrix
    evaluator_rf.plot_confusion_matrix(X_test, y_test)

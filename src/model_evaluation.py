import pickle
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
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
from temporal_cnn import TCNN


class ModelEvaluator:
    def __init__(self, model, model_type: str, batch_size: int = 32):
        """
        Parameters:
        -----------
        model : Union[RF model, TCNN]
            The model to evaluate
        model_type : str
            Either 'RF' or 'torch'
        batch_size : int
            Batch size for PyTorch model evaluation
        """
        self.model = model
        self.model_type = model_type
        self.batch_size = batch_size
        if model_type == "TCNN":
            self.device = model.device

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Unified prediction method for both model types"""
        if self.model_type == "RF":
            return self.model.predict(X)
        else:
            # Create dataset and dataloader for batched predictions
            test_dataset = TensorDataset(torch.FloatTensor(X))
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

            # Lists to store predictions
            y_pred_list = []

            # Set model to evaluation mode
            self.model.eval()

            # Iterate through batches
            with torch.no_grad():
                for (X_batch,) in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_pred_batch = self.model(X_batch)
                    y_pred_list.append(y_pred_batch.cpu().numpy())

            # Concatenate all batches
            y_pred = np.concatenate(y_pred_list).squeeze()
            return (y_pred > 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Unified evaluation method"""
        if self.model_type == "RF":
            y_pred = self.predict(X)
        else:
            # Create dataset and dataloader for batched predictions
            test_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

            # Lists to store predictions and true labels
            y_pred_list = []
            y_true_list = []

            # Set model to evaluation mode
            self.model.eval()

            # Iterate through batches
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_pred_batch = self.model(X_batch)
                    y_pred_list.append(y_pred_batch.cpu().numpy())
                    y_true_list.append(y_batch.numpy())

            # Concatenate all batches
            y_pred = np.concatenate(y_pred_list).squeeze()
            y_true = np.concatenate(y_true_list)
            y_pred = (y_pred > 0.5).astype(int)

        y_true = y.astype(int)

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

    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray, ax=None) -> None:
        """Plot confusion matrix with optional axis specification"""
        y_pred = self.predict(X)
        y_true = y.astype(int)

        cm = confusion_matrix(y_true, y_pred)
        if ax is None:
            plt.figure(figsize=(6, 4))
            ax = plt.gca()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Event", "Extreme Event"],
            yticklabels=["No Event", "Extreme Event"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {self.model_type}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")


def compare_models(evaluators: dict, data_dict: dict):
    """
    Compare multiple models using various metrics and visualizations.

    Parameters:
    -----------
    evaluators : dict
        Dictionary of model evaluators with format {model_name: evaluator}
    data_dict : dict
        Dictionary of data with format {model_name: (X_test, y_test)}
    """
    # Collect metrics for all models
    metrics_dict = {}
    for name, evaluator in evaluators.items():
        X_test, y_test = data_dict[name]
        metrics_dict[name] = evaluator.evaluate(X_test, y_test)

    # Print comparison table
    print("\nModel Comparison:")
    print("-" * 60)

    # Print header with model names
    header = f"{'Metric':20s}"
    for name in evaluators.keys():
        header += f" {name:>15s}"
    print(header)
    print("-" * 60)

    # Print metrics
    for metric in metrics_dict[list(metrics_dict.keys())[0]].keys():
        row = f"{metric:20s}"
        for name in evaluators.keys():
            row += f" {metrics_dict[name][metric]:15.4f}"
        print(row)
    print("-" * 60)

    # Plot confusion matrices side by side
    n_models = len(evaluators)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, evaluator) in zip(axes, evaluators.items()):
        X_test, y_test = data_dict[name]
        evaluator.plot_confusion_matrix(X_test, y_test, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

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

    # Store original array format for NN
    X_test_seq, y_test_seq = preprocessor.create_sequences(X_test, y_test, lookback=10)

    # Convert time series data into supervised learning format
    X_train, y_train = preprocessor.time_series_to_supervised(
        X_train, y_train, lookback=10
    )
    X_val, y_val = preprocessor.time_series_to_supervised(X_val, y_val, lookback=10)
    X_test, y_test = preprocessor.time_series_to_supervised(X_test, y_test, lookback=10)

    # Initialize evaluators
    rf_evaluator = ModelEvaluator(
        model=pickle.load(open("models/best_random_forest.pkl", "rb")),
        model_type="RF",
    )

    nn = TCNN.load_from_checkpoint(
        "checkpoints/best_model.ckpt", n_features=X_test_seq.shape[1], lookback=10
    )
    nn_evaluator = ModelEvaluator(model=nn, model_type="TCNN")

    # Prepare evaluators and data dictionaries
    evaluators = {"Random Forest": rf_evaluator, "Neural Network": nn_evaluator}

    data_dict = {
        "Random Forest": (X_test, y_test),
        "Neural Network": (X_test_seq, y_test_seq),
    }

    # Compare models
    compare_models(evaluators, data_dict)

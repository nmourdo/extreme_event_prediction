"""Model evaluation utilities for time series classification models.

This module provides tools for evaluating and comparing different machine learning models
(Random Forest, TCNN, LSTM) trained for time series classification tasks.
It focuses on metrics particularly relevant for imbalanced data and extreme event prediction.

Classes:
    ModelEvaluator: Unified interface for evaluating different model types

Features:
    - Support for multiple model architectures:
        * Random Forest
        * Temporal CNN (TCNN)
        * LSTM
    - Comprehensive evaluation metrics:
        * F2 Score 
        * F1 Score
        * Precision and Recall
        * ROC AUC
        * Balanced Accuracy
    - Visualization utilities:
        * Confusion matrices
        * Side-by-side model comparisons

Notes:
    - Different models may require different input shapes:
        * Random Forest: 2D array (samples, features)
        * TCNN: 3D array (samples, features, timesteps)
        * LSTM: 3D array (samples, timesteps, features)
    - Metrics like balanced accuracy and AUC account for class imbalance.
    
"""

import pickle
from typing import Union

import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
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
from improvement import LSTM


class ModelEvaluator:
    def __init__(
        self,
        model: Union[RandomForestClassifier, TCNN, LSTM],
        model_type: str,
        batch_size: int = 32,
    ):
        """Initialize a ModelEvaluator instance.

        Parameters
        ----------
        model : Union[RandomForestClassifier, TCNN, LSTM]
            The model to evaluate. Can be either a Random Forest classifier, a Temporal CNN model or an LSTM model.
        model_type : str
            The type of model being evaluated. Must be either 'RF' for Random Forest, 'TCNN' for Temporal CNN or 'LSTM' for LSTM.
        batch_size : int, optional
            Batch size for PyTorch model evaluation, by default 32. Only used when model_type is 'TCNN' or 'LSTM'.
        """
        self.model = model
        self.model_type = model_type
        self.batch_size = batch_size
        # Set device for all PyTorch models
        if model_type in ["TCNN", "LSTM", "MLP"]:
            self.device = next(model.parameters()).device
        else:
            self.device = "cpu"  # Default device for non-PyTorch models

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the model.

        Parameters
        ----------
        X : np.ndarray
            Input features to make predictions on.
            For Random Forest, this should be a 2D array.
            For TCNN this should be a 3D array with shape (samples, features, timesteps).
            For LSTM this should be a 3D array with shape (samples, timesteps, features).

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1) for each input sample.
        """
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
        """Evaluate the model's performance using multiple metrics.

        Parameters
        ----------
        X : np.ndarray
            Input features to make predictions on.
            For Random Forest, this should be a 2D array.
            For TCNN this should be a 3D array with shape (samples, features, timesteps).
            For LSTM this should be a 3D array with shape (samples, timesteps, features).
        y : np.ndarray
            True labels (0 or 1) for each input sample.

        Returns
        -------
        dict
            Dictionary containing various performance metrics:
            - F2 Score: F-beta score with beta=2
            - F1 Score: Standard F1 score
            - Precision: Precision score
            - Recall: Recall score
            - Accuracy: Standard accuracy
            - AUC: Area Under the ROC Curve
            - Balanced Accuracy: Accuracy that accounts for class imbalance
        """
        if self.model_type == "RF":
            y_pred = self.predict(X)
        else:
            # Create dataset and dataloader for batched predictions
            test_dataset = TensorDataset(
                torch.FloatTensor(X).to(self.device),
                torch.FloatTensor(y).to(self.device),
            )
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
                    y_pred_batch = self.model(X_batch)
                    y_pred_list.append(y_pred_batch.cpu().numpy())
                    y_true_list.append(y_batch.cpu().numpy())

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
        """Plot a confusion matrix for the model's predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features. For Random Forest, this should be a 2D array.
            For TCNN, this should be a 3D array with shape (samples, timesteps, features).
        y : np.ndarray
            True labels (0 or 1) for each input sample.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the confusion matrix. If None, a new figure is created.

        Returns
        -------
        None
            The function displays the confusion matrix plot but doesn't return anything.
        """
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
        plt.tight_layout()
        if ax is None:
            plt.show()


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

    # Prepare test data for random forest and initialize evaluator
    preprocessor = StockDataPreprocessor(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-31"
    )
    stock_prices = preprocessor.download_and_prepare_stock_data()

    _, _, _, _, X_test, y_test = preprocessor.split_data(
        stock_prices,
        ["Open", "High", "Low", "Close", "Volume", "Daily_Returns"],
        "Extreme_Event",
        train_ratio=0.7,
        val_ratio=0.85,
    )
    X_test, y_test = preprocessor.time_series_to_supervised(X_test, y_test, lookback=10)

    rf_evaluator = ModelEvaluator(
        model=pickle.load(open("./data/models/best_random_forest.pkl", "rb")),
        model_type="RF",
    )

    # Prepare test data for neural network and initialize evaluator
    stock_prices_standardized = preprocessor.standardize_data(
        stock_prices, ["Open", "High", "Low", "Close", "Volume"]
    )
    _, _, _, _, X_test_seq, y_test_seq = preprocessor.split_data(
        stock_prices_standardized,
        ["Open", "High", "Low", "Close", "Volume", "Daily_Returns"],
        "Extreme_Event",
        train_ratio=0.7,
        val_ratio=0.85,
    )
    X_test_seq, y_test_seq = preprocessor.create_sequences(
        X_test_seq,
        y_test_seq,
        lookback=10,
    )

    nn = TCNN(
        n_features=X_test_seq.shape[1],
        lookback=10,
        hidden_dim=128,
        conv_channels=64,
        kernel_size=3,
        dropout_prob=0.3,
        learning_rate=1e-4,
    )
    nn.load_state_dict(torch.load("./data/models/best_tcnn.pth"))
    nn_evaluator = ModelEvaluator(model=nn, model_type="TCNN")

    # Prepare evaluators and data dictionaries
    evaluators = {"Random Forest": rf_evaluator, "Neural Network": nn_evaluator}

    data_dict = {
        "Random Forest": (X_test, y_test),
        "Neural Network": (X_test_seq, y_test_seq),
    }

    # Compare models
    compare_models(evaluators, data_dict)

import random
import warnings
from typing import Any, Dict, Tuple, Type

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from ray.tune.search.hyperopt import HyperOptSearch

# Local imports
from data_preprocessing import StockDataPreprocessor
from model_evaluation import ModelEvaluator
from random_forest import RandomForestOptimizer

# Suppress all warnings
warnings.filterwarnings("ignore")


class BaseModel(L.LightningModule):
    """Base model class implementing common functionality for deep learning models.

    This class provides the basic training, validation and optimization setup using PyTorch Lightning.
    It implements weighted binary cross-entropy loss to handle class imbalance during training.

    Args:
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
    """

    def __init__(self, learning_rate: float = 1e-4) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Implements weighted binary cross-entropy loss to handle class imbalance.
        Weights are calculated based on class distribution in each batch.

        Args:
            batch (tuple): Tuple containing input features and target labels
            batch_idx (int): Index of current batch

        Returns:
            torch.Tensor: Training loss for current batch
        """
        x, y = batch
        y_hat = self(x)

        # Calculate weights for the loss function
        n_samples = len(y)
        n_positive = max(y.sum().item(), 1e-7)  # Avoid division by zero
        n_negative = n_samples - n_positive

        # Create sample weights with clamping to avoid extreme values
        pos_weight = min(n_samples / (2 * n_positive), 10.0)  # Clamp maximum weight
        neg_weight = n_samples / (2 * n_negative)
        weights = torch.where(
            y.view(-1, 1) == 1, torch.tensor(pos_weight), torch.tensor(neg_weight)
        )

        # Use weighted BCE loss
        criterion = nn.BCELoss(weight=weights.to(self.device))
        loss = criterion(y_hat, y.float().view(-1, 1))

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step.

        Args:
            batch (tuple): Tuple containing input features and target labels
            batch_idx (int): Index of current batch

        Returns:
            torch.Tensor: Validation loss for current batch
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for training.
            Adam with L2 penalty is used as a countermeasure against overfitting.

        Returns:
            torch.optim.Optimizer: AdamW optimizer instance
        """
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )


class FNN(BaseModel):
    """Feed-forward neural network model for binary classification.

    Args:
        n_features (int): Number of input features
        hidden_dim (int, optional): Dimension of hidden layers. Defaults to 128.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.3.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        dropout_prob: float = 0.3,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__(learning_rate)
        self.fnn = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features)

        Returns:
            torch.Tensor: Model predictions of shape (batch_size, 1)
        """
        return torch.sigmoid(self.fnn(x))


class LSTM(BaseModel):
    """Long Short-Term Memory (LSTM) model for binary classification.

    Args:
        n_features (int): Number of input features per timestep
        lookback (int): Number of timesteps to look back
        hidden_dim (int, optional): Dimension of LSTM hidden state. Defaults to 256.
        num_layers (int, optional): Number of LSTM layers. Defaults to 2.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.2.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 3e-4.
    """

    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        learning_rate: float = 3e-4,
    ) -> None:
        super().__init__(learning_rate)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, lookback, n_features)

        Returns:
            torch.Tensor: Model predictions of shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return torch.sigmoid(self.classifier(last_output))


def train_nn(
    model: L.LightningModule,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Train a NN model.

    Args:
        model: The model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
       None
    """
    trainer = L.Trainer(
        max_epochs=1, callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
    )
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RandomForestClassifier:
    """
    Train and evaluate a Random Forest model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        A tuple containing the evaluation metrics and the confusion matrix.
    """
    rf_optimizer = RandomForestOptimizer()
    rf_optimizer.optimize(X_train, y_train, X_val, y_val)
    final_model = rf_optimizer.train_best_model(X_train, y_train)
    # evaluator = ModelEvaluator(model=final_model, model_type="RF")
    # metrics = evaluator.evaluate(X_test, y_test)
    # confusion_matrix = evaluator.plot_confusion_matrix(X_test, y_test)
    return final_model


def tune_model(
    config: Dict[str, Any],
    model_class: Type[BaseModel],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """
    Tune a model using Ray Tune.

    Args:
        config: The hyperparameter configuration.
        model_class: The class of the model to tune.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
    """
    # Determine the model parameters based on the model class
    model_params = {
        "n_features": X_train.shape[1] if model_class == FNN else X_train.shape[2],
        "hidden_dim": config["hidden_dim"],
        "dropout_prob": config["dropout_prob"],
        "learning_rate": config["learning_rate"],
    }

    # Add lookback and num_layers only for LSTM
    if model_class == LSTM:
        model_params["lookback"] = X_train.shape[1]
        model_params["num_layers"] = config.get("num_layers", 1)

    # Instantiate the model
    model = model_class(**model_params)

    trainer = L.Trainer(
        max_epochs=1,
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],
        enable_progress_bar=False,
    )
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    preprocessor = StockDataPreprocessor(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-31"
    )
    stock_prices = preprocessor.download_and_prepare_stock_data()
    stock_prices = preprocessor.add_features()
    stock_prices = preprocessor.standardize_data(
        stock_prices,
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "rolling_volatility",
            "relative_volume",
            "VIX",
            "bollinger_band_width",
        ],
    )

    X_train, y_train, X_val, y_val, X_test, y_test = StockDataPreprocessor.split_data(
        stock_prices,
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "rolling_volatility",
            "relative_volume",
            "VIX",
            "bollinger_band_width",
            "ATR",
            "Daily_Returns",
        ],
        "Extreme_Event",
        train_ratio=0.7,
        val_ratio=0.85,
    )

    # Prepare data for FNN and random forest
    X_train_fnn, y_train_fnn = StockDataPreprocessor.time_series_to_supervised(
        X_train, y_train, lookback=10
    )
    X_val_fnn, y_val_fnn = StockDataPreprocessor.time_series_to_supervised(
        X_val, y_val, lookback=10
    )
    X_test_fnn, y_test_fnn = StockDataPreprocessor.time_series_to_supervised(
        X_test, y_test, lookback=10
    )

    # Prepare data for LSTM
    X_train_lstm, y_train_lstm = preprocessor.create_sequences(
        X_train, y_train, lookback=10
    )
    X_val_lstm, y_val_lstm = preprocessor.create_sequences(X_val, y_val, lookback=10)
    X_test_lstm, y_test_lstm = preprocessor.create_sequences(
        X_test, y_test, lookback=10
    )

    # Transpose the data to match the expected input shape for LSTM
    X_train_lstm = np.transpose(X_train_lstm, (0, 2, 1))
    X_val_lstm = np.transpose(X_val_lstm, (0, 2, 1))
    X_test_lstm = np.transpose(X_test_lstm, (0, 2, 1))

    # Define hyperparameter search spaces
    fnn_space = {
        "hidden_dim": tune.choice([64, 128, 256]),
        "dropout_prob": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
    }

    lstm_space = {
        "hidden_dim": tune.choice([64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout_prob": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
    }

    # Initialize Ray
    ray.init(log_to_driver=False)

    # Tune FNN model
    print("Tuning the FNN model...")
    fnn_search = HyperOptSearch(space=fnn_space, metric="loss", mode="min")
    fnn_analysis = tune.run(
        tune.with_parameters(
            tune_model,
            model_class=FNN,
            X_train=X_train_fnn.values,
            y_train=y_train_fnn.values,
            X_val=X_val_fnn.values,
            y_val=y_val_fnn.values,
        ),
        num_samples=20,
        scheduler=ASHAScheduler(max_t=10, grace_period=10, reduction_factor=2),
        search_alg=fnn_search,
        metric="loss",
        mode="min",
        verbose=0,
    )
    best_fnn_config = fnn_analysis.get_best_config(metric="loss", mode="min")
    print("Best FNN hyperparameters:", best_fnn_config)
    print("Training the FNN model...")
    best_fnn_model = FNN(
        n_features=X_train_fnn.shape[1],
        hidden_dim=best_fnn_config["hidden_dim"],
        dropout_prob=best_fnn_config["dropout_prob"],
        learning_rate=best_fnn_config["learning_rate"],
    )
    train_nn(
        best_fnn_model,
        X_train_fnn.values,
        y_train_fnn.values,
        X_val_fnn.values,
        y_val_fnn.values,
    )
    fnn_evaluator = ModelEvaluator(model=best_fnn_model, model_type="FNN")
    fnn_metrics = fnn_evaluator.evaluate(X_test_fnn.values, y_test_fnn.values)

    # Tune LSTM model
    print("Tuning the LSTM model...")
    lstm_search = HyperOptSearch(space=lstm_space, metric="loss", mode="min")
    lstm_analysis = tune.run(
        tune.with_parameters(
            tune_model,
            model_class=LSTM,
            X_train=X_train_lstm,
            y_train=y_train_lstm,
            X_val=X_val_lstm,
            y_val=y_val_lstm,
        ),
        num_samples=20,
        scheduler=ASHAScheduler(max_t=10, grace_period=10, reduction_factor=2),
        search_alg=lstm_search,
        metric="loss",
        mode="min",
        verbose=0,
    )
    # Get the best hyperparameters
    best_lstm_config = lstm_analysis.get_best_config(metric="loss", mode="min")
    print("Best LSTM hyperparameters:", best_lstm_config)
    print("Training the LSTM model...")
    best_lstm_model = LSTM(
        n_features=X_train_lstm.shape[2],
        lookback=X_train_lstm.shape[1],
        hidden_dim=best_lstm_config["hidden_dim"],
        num_layers=best_lstm_config["num_layers"],
        dropout_prob=best_lstm_config["dropout_prob"],
        learning_rate=best_lstm_config["learning_rate"],
    )
    train_nn(
        best_lstm_model,
        X_train_lstm,
        y_train_lstm,
        X_val_lstm,
        y_val_lstm,
    )
    lstm_evaluator = ModelEvaluator(model=best_lstm_model, model_type="LSTM")
    lstm_metrics = lstm_evaluator.evaluate(X_test_lstm, y_test_lstm)

    # Train and evaluate Random Forest
    print("Training the Random Forest model...")
    best_rf_model = train_random_forest(X_train_fnn, y_train_fnn, X_val_fnn, y_val_fnn)
    rf_evaluator = ModelEvaluator(model=best_rf_model, model_type="RF")
    rf_metrics = rf_evaluator.evaluate(X_test_fnn, y_test_fnn)

    # Compare the models
    print("\nModel Comparison:")
    print(f"{'Metric':<20} {'FNN':<10} {'LSTM':<10} {'Random Forest':<10}")
    for metric in fnn_metrics.keys():
        print(
            f"{metric:<20} {fnn_metrics[metric]:<10.4f} {lstm_metrics[metric]:<10.4f} {rf_metrics[metric]:<10.4f}"
        )

    # Create a figure with 3 subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Plot confusion matrices
    fnn_evaluator.plot_confusion_matrix(X_test_fnn.values, y_test_fnn.values, ax=ax1)

    lstm_evaluator.plot_confusion_matrix(X_test_lstm, y_test_lstm, ax=ax2)

    rf_evaluator.plot_confusion_matrix(X_test_fnn, y_test_fnn, ax=ax3)

    plt.tight_layout()
    plt.show()

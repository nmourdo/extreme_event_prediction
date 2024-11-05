import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

try:
    from src.data_preprocessing import StockDataPreprocessor
except ModuleNotFoundError:
    from data_preprocessing import StockDataPreprocessor


class LSTMClassifier:
    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_prob: float = 0.3,
        learning_rate: float = 1e-4,
    ) -> None:
        """
        Classifier class for LSTM model.

        Args:
            n_features: Number of input features
            lookback: Number of timesteps to look back
            hidden_dim: Dimension of hidden layer
            num_layers: Number of LSTM layers
            dropout_prob: Dropout probability
            learning_rate: Learning rate for optimizer
        """
        self.model = LSTM(
            n_features=n_features,
            lookback=lookback,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            learning_rate=learning_rate,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
    ) -> ModelCheckpoint:
        """Train the model with early stopping and model checkpointing."""
        # Numpy arrays to torch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Early stopping and best model checkpointing callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"best_{self.model.__class__.__name__.lower()}_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )

        logger = L.pytorch.loggers.CSVLogger(
            "logs", name=self.model.__class__.__name__.lower()
        )

        # Trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            accelerator="auto",
            devices=1,
            logger=logger,
        )

        # Train model
        trainer.fit(
            self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        return checkpoint_callback

    def plot_training_history(self) -> None:
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))

        # Plot raw losses
        plt.plot(
            range(len(self.model.train_losses)),
            self.model.train_losses,
            "b-",
            label="Training Loss",
        )

        # Adjust validation loss x-axis to match actual validation frequency
        val_freq = len(self.model.train_losses) / len(self.model.val_losses)
        val_x = [int(i * val_freq) for i in range(len(self.model.val_losses))]
        plt.plot(val_x, self.model.val_losses, "r-", label="Validation Loss")

        plt.title("Training and Validation Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


class LSTM(L.LightningModule):
    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        learning_rate: float = 3e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []

        # Standard LSTM with increased capacity
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True,
        )

        # Improved classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM expects input of shape (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]

        # Pass through classifier layers and add sigmoid activation
        return torch.sigmoid(self.classifier(last_output))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of current batch

        Returns:
            Training loss value
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

        # Store training loss
        self.train_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single validation step.

        Args:
            batch: Tuple of (features, labels)
            batch_idx: Index of current batch

        Returns:
            Validation loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y.float().view(-1, 1))
        # Store validation loss
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            Optimizer instance
        """
        # Add weight decay for regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,  # L2 regularization
        )
        return optimizer


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Insantiate the preprocessor and download the data
    preprocessor = StockDataPreprocessor(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-31"
    )
    stock_prices = preprocessor.download_and_prepare_stock_data()
    # Add 10-day rolling volatility, volume relative to 10-day moving average and VIX index
    stock_prices = preprocessor.add_features()

    # Standardize the features except for the daily returns
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

    # Split data into train, validation and test sets
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

    # Convert the features into sequences
    X_train, y_train = preprocessor.create_sequences(X_train, y_train, lookback=10)
    X_val, y_val = preprocessor.create_sequences(X_val, y_val, lookback=10)
    X_test, y_test = preprocessor.create_sequences(X_test, y_test, lookback=10)

    # Transpose the data to match the expected input shape for LSTM
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    # Train the model
    lstm_classifier = LSTMClassifier(
        n_features=X_train.shape[2],
        lookback=X_train.shape[1],
        hidden_dim=128,
        num_layers=3,
        dropout_prob=0.5,
        learning_rate=1e-3,
    )
    checkpoint_callback = lstm_classifier.train(
        X_train, y_train, X_val, y_val, batch_size=32, max_epochs=100, patience=20
    )

    # Plot training history
    lstm_classifier.plot_training_history()

    # Evaluate the model
    from model_evaluation import ModelEvaluator

    lstm_evaluator = ModelEvaluator(
        model=lstm_classifier.model,
        model_type="LSTM",
    )
    metrics = lstm_evaluator.evaluate(X_test, y_test)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    print("-" * 40)

    lstm_evaluator.plot_confusion_matrix(X_test, y_test)

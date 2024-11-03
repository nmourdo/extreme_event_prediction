import numpy as np
import random
import lightning as L
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.data_preprocessing import StockDataPreprocessor
except ModuleNotFoundError:
    from data_preprocessing import StockDataPreprocessor


class TCNN(L.LightningModule):
    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 128,
        conv_channels: int = 64,
        kernel_size: int = 3,
        dropout_prob: float = 0.3,
        learning_rate: float = 1e-4,
    ):
        """
        Parameters:
        -----------
        n_features : int
            Number of input features
        lookback : int
            Number of timesteps to look back
        hidden_dim : int
            Dimension of hidden dense layer
        conv_channels : int
            Number of channels in first conv layer (second layer has 2x channels)
        kernel_size : int
            Size of convolutional kernel
        learning_rate : float
            Learning rate for optimizer
        """
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []

        # Definition of the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=conv_channels,
                kernel_size=kernel_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Conv1d(
                in_channels=conv_channels,
                out_channels=conv_channels * 2,
                kernel_size=kernel_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Flatten(),
            nn.Linear(
                (conv_channels * 2) * (lookback - 2 * kernel_size + 2), hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, n_features, lookback]

        Returns:
            Predicted probabilities tensor of shape [batch_size, 1]
        """
        x = self.cnn(x)
        return torch.sigmoid(x)

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
        loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
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
        loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        # Store validation loss
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            AdamW optimizer instance
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class ExtremeEventNNPredictor:
    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 64,
        conv_channels: int = 32,
        kernel_size: int = 3,
        dropout_prob: float = 0.2,
        learning_rate: float = 1e-3,
    ) -> None:
        self.model = TCNN(
            n_features=n_features,
            lookback=lookback,
            hidden_dim=hidden_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
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
        """Train the CNN model with early stopping and model checkpointing."""
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
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )

        logger = L.pytorch.loggers.CSVLogger("logs", name="tcnn")

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
    # Standardize the features except for the daily returns,
    # which is already in a similar scale
    stock_prices = preprocessor.standardize_data(
        stock_prices, ["Open", "High", "Low", "Close", "Volume"]
    )

    # Split data into train, validation and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(
        stock_prices,
        ["Open", "High", "Low", "Close", "Volume", "Daily_Returns"],
        "Extreme_Event",
        train_ratio=0.7,
        val_ratio=0.85,
    )

    # Convert the features into sequences
    X_train, y_train = preprocessor.create_sequences(X_train, y_train, lookback=10)
    X_val, y_val = preprocessor.create_sequences(X_val, y_val, lookback=10)
    X_test, y_test = preprocessor.create_sequences(X_test, y_test, lookback=10)

    # Train the model
    predictor = ExtremeEventNNPredictor(
        n_features=X_train.shape[1],
        lookback=X_train.shape[2],
        hidden_dim=64,
        conv_channels=96,
        kernel_size=3,
        dropout_prob=0.3,
        learning_rate=1e-4,
    )
    checkpoint_callback = predictor.train(
        X_train, y_train, X_val, y_val, batch_size=32, max_epochs=100, patience=20
    )

    # Plot training history
    predictor.plot_training_history()

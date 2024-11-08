"""Temporal Convolutional Neural Network implementation for time series classification.

This module provides implementations of TCNN (Temporal Convolutional Neural Network)
models for binary classification of financial time series data. It includes:

Classes:
    TCNNClassifier: High-level wrapper for training and visualization
    TCNN: PyTorch Lightning implementation of the neural network

Features:
    - PyTorch Lightning for hardware acceleration and training utilities
    - Class imbalance handling through weighted loss
    - Automatic learning rate adjustment using ReduceLROnPlateau
    - Early stopping and model checkpointing
    - Hyperparameters of the best model and training history are saved to CSV files automatically
    - Training visualization utilities

Note:
    - The best performing TCNN was not returned consistently along multiple runs
    of the code (despite fixining all possible random seeds). For this reason,
    the best performing model was saved and is used for the final evaluation in the
    `model_evaluation.py` module.

"""

import random

import numpy as np
import lightning as L
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import StockDataPreprocessor

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
L.seed_everything(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TCNNClassifier:
    """Temporal Convolutional Neural Network (TCNN) Classifier for time series data.

    This class provides a wrapper around the TCNN model with utilities for training
    and visualization. It uses PyTorch Lightning for training with automatic
    hardware acceleration, early stopping, and model checkpointing.

    Args:
        n_features (int): Number of input features
        lookback (int): Number of timesteps to look back
        hidden_dim (int, optional): Dimension of hidden feed-forward layer. Defaults to 128.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.3.
        conv_channels (int, optional): Number of channels in first conv layer. Defaults to 64.
        kernel_size (int, optional): Size of convolutional kernel. Defaults to 3.
        scheduler_patience (int, optional): Epochs before learning rate reduction. Defaults to 5.
        scheduler_factor (float, optional): Factor to reduce learning rate. Defaults to 0.5.
        min_lr (float, optional): Minimum learning rate. Defaults to 1e-6.

    Attributes:
        model (TCNN): The underlying TCNN model instance

    Example:
        >>> classifier = TCNNClassifier(n_features=6, lookback=10)
        >>> checkpoint = classifier.train(X_train, y_train, X_val, y_val)
        >>> classifier.plot_training_history()
    """

    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        dropout_prob: float = 0.3,
        conv_channels: int = 64,
        kernel_size: int = 3,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
    ) -> None:
        """
        Classifier class for TCNN model.

        Args:
            n_features: Number of input features
            lookback: Number of timesteps to look back
            hidden_dim: Dimension of hidden feed-forward layer
            learning_rate: Learning rate for optimizer
            dropout_prob: Dropout probability
            conv_channels: Number of channels in first conv layer
            kernel_size: Size of convolutional kernel
            scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced
            scheduler_factor: Factor by which the learning rate will be reduced
            min_lr: Minimum learning rate

        """
        self.model = TCNN(
            n_features=n_features,
            lookback=lookback,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout_prob=dropout_prob,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            min_lr=min_lr,
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
        """Train the model with early stopping and model checkpointing.

        Args:
            X_train (np.ndarray): Training features of shape (n_samples, n_features, lookback)
            y_train (np.ndarray): Training labels of shape (n_samples,)
            X_val (np.ndarray): Validation features of shape (n_samples, n_features, lookback)
            y_val (np.ndarray): Validation labels of shape (n_samples,)
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            max_epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to 10.

        Returns:
            ModelCheckpoint: Callback containing information about the best saved model

        Notes:
            - The model is automatically saved to 'checkpoints' directory
            - Training logs are saved to 'logs' directory
        """
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
            save_last=False,
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
            log_every_n_steps=10,
        )

        # Train model
        trainer.fit(
            self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        return checkpoint_callback

    def plot_training_history(self) -> None:
        """Plot the training and validation loss history.

        Creates a matplotlib figure showing:
        - Training loss curve in blue
        - Validation loss curve in red
        - Both curves aligned to show actual validation frequency
        - Grid and legend for better readability

        The plot is automatically displayed using plt.show()

        Raises:
            AttributeError: If called before training the model
        """
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


class TCNN(L.LightningModule):
    """Temporal Convolutional Neural Network (TCNN) implementation.

    A TCNN for binary classification of time series data.
    The architecture consists of two 1D convolutional layers followed by dense layers,
    with ReLU activations and dropout for regularization.

    Args:
        n_features (int): Number of input features
        lookback (int): Number of timesteps to look back
        hidden_dim (int, optional): Dimension of hidden dense layer. Defaults to 128.
        conv_channels (int, optional): Number of channels in first conv layer. Defaults to 64.
        kernel_size (int, optional): Size of convolutional kernel. Defaults to 3.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.3.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        scheduler_patience (int, optional): Epochs before learning rate reduction. Defaults to 5.
        scheduler_factor (float, optional): Factor to reduce learning rate. Defaults to 0.5.
        min_lr (float, optional): Minimum learning rate. Defaults to 1e-6.

    Attributes:
        learning_rate (float): Current learning rate
        scheduler_patience (int): Patience for learning rate scheduler
        scheduler_factor (float): Reduction factor for learning rate
        min_lr (float): Minimum learning rate threshold
        train_losses (list): History of training losses
        val_losses (list): History of validation losses
        cnn (nn.Sequential): The neural network architecture
    """

    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 128,
        conv_channels: int = 64,
        kernel_size: int = 3,
        dropout_prob: float = 0.3,
        learning_rate: float = 1e-4,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
    ) -> None:
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
        scheduler_patience : int
            Number of epochs with no improvement after which learning rate will be reduced
        scheduler_factor : float
            Factor by which the learning rate will be reduced
        min_lr : float
            Minimum learning rate
        """
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.min_lr = min_lr
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
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features, lookback)

        Returns:
            torch.Tensor: Model predictions of shape (batch_size, 1) after sigmoid activation
        """
        x = self.cnn(x)
        return torch.sigmoid(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Implements weighted BCE loss to handle class imbalance, with weight clamping
        to prevent extreme values.

        Args:
            batch (tuple): Tuple of (features, labels)
            batch_idx (int): Index of current batch

        Returns:
            torch.Tensor: Weighted binary cross-entropy loss

        Notes:
            - Automatically logs training loss
            - Implements dynamic class weighting based on batch statistics
            - Clamps positive class weight to maximum of 10.0
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

        criterion = nn.BCELoss(weight=weights.to(self.device))
        loss = criterion(y_hat, y.float().view(-1, 1))

        self.train_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step.

        Args:
            batch (tuple): Tuple of (features, labels)
            batch_idx (int): Index of current batch

        Returns:
            torch.Tensor: Binary cross-entropy loss

        Notes:
            - Automatically logs validation loss at epoch level
            - Uses unweighted BCE loss for validation (unlike training) to get
              unbiased performance estimates on real-world data
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict:
        """Configures the optimizer and learning rate scheduler.

        Sets up AdamW optimizer with weight decay and ReduceLROnPlateau scheduler
        for learning rate adjustment based on validation loss.

        Returns:
            dict: Configuration containing:
                - optimizer: AdamW optimizer with weight decay
                - lr_scheduler: Dict with scheduler configuration
                    - scheduler: ReduceLROnPlateau instance
                    - monitor: Metric to monitor ('val_loss')
                    - frequency: Update frequency (1)

        Notes:
            - Scheduler reduces learning rate when validation loss plateaus
            - Learning rate will not go below min_lr
        """
        # Add weight decay for regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.min_lr,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


if __name__ == "__main__":

    # Initialize data preprocessor and load data
    preprocessor = StockDataPreprocessor(
        ticker="AAPL", start_date="2015-01-01", end_date="2024-01-31"
    )
    stock_prices = preprocessor.download_and_prepare_stock_data()
    # Standardize the features except for the daily returns, which is
    # already in a similar scale
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

    # Convert the features into sequences of shape (n_samples, n_features, n_lookback)
    X_train, y_train = preprocessor.create_sequences(X_train, y_train, lookback=10)
    X_val, y_val = preprocessor.create_sequences(X_val, y_val, lookback=10)
    X_test, y_test = preprocessor.create_sequences(X_test, y_test, lookback=10)

    # Train the model
    tcnn_classifier = TCNNClassifier(
        n_features=X_train.shape[1],
        lookback=X_train.shape[2],
        hidden_dim=64,
        conv_channels=96,
        kernel_size=3,
        dropout_prob=0.3,
        learning_rate=1e-4,
        scheduler_patience=5,
        scheduler_factor=0.5,
        min_lr=1e-6,
    )
    print("-" * 50)
    print("Training TCNN model...")
    print("-" * 50)
    checkpoint_callback = tcnn_classifier.train(
        X_train, y_train, X_val, y_val, batch_size=32, max_epochs=100, patience=20
    )

    # Plot training history
    tcnn_classifier.plot_training_history()

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from data_preprocessing import StockDataPreprocessor

import numpy as np
import random
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import StockDataPreprocessor


class FNNClassifier:
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        dropout_prob: float = 0.3,
    ) -> None:
        self.model = FNN(
            n_features=n_features,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout_prob=dropout_prob,
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
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            accelerator="auto",
            devices=1,
            logger=logger,
        )

        trainer.fit(
            self.model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        return checkpoint_callback

    def plot_training_history(self) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(self.model.train_losses)),
            self.model.train_losses,
            "b-",
            label="Training Loss",
        )
        val_freq = len(self.model.train_losses) / len(self.model.val_losses)
        val_x = [int(i * val_freq) for i in range(len(self.model.val_losses))]
        plt.plot(val_x, self.model.val_losses, "r-", label="Validation Loss")

        plt.title("Training and Validation Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


class FNN(L.LightningModule):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        dropout_prob: float = 0.3,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []

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
        x = self.fnn(x)
        return torch.sigmoid(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        n_samples = len(y)
        n_positive = max(y.sum().item(), 1e-7)
        n_negative = n_samples - n_positive

        pos_weight = min(n_samples / (2 * n_positive), 10.0)
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
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        return {"optimizer": optimizer}


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

    X_train, y_train = StockDataPreprocessor.time_series_to_supervised(
        X_train, y_train, lookback=10
    )
    X_val, y_val = StockDataPreprocessor.time_series_to_supervised(
        X_val, y_val, lookback=10
    )
    X_test, y_test = StockDataPreprocessor.time_series_to_supervised(
        X_test, y_test, lookback=10
    )

    fnn_classifier = FNNClassifier(
        n_features=X_train.shape[1],
        hidden_dim=128,
        dropout_prob=0.3,
        learning_rate=1e-4,
    )
    checkpoint_callback = fnn_classifier.train(
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values,
        batch_size=32,
        max_epochs=100,
        patience=20,
    )

    fnn_classifier.plot_training_history()

    from model_evaluation import ModelEvaluator

    fnn_evaluator = ModelEvaluator(
        model=fnn_classifier.model,
        model_type="FNN",
    )

    metrics = fnn_evaluator.evaluate(X_test.values, y_test.values)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    print("-" * 40)

    # Plot the confusion matrix
    fnn_evaluator.plot_confusion_matrix(X_test.values, y_test.values)

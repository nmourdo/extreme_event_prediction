import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.data_preprocessing import StockDataPreprocessor
    from src.random_forest import RandomForestOptimizer
except ModuleNotFoundError:
    from data_preprocessing import StockDataPreprocessor
    from random_forest import RandomForestOptimizer


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

    X_train, y_train = StockDataPreprocessor.time_series_to_supervised(
        X_train, y_train, lookback=10
    )
    X_val, y_val = StockDataPreprocessor.time_series_to_supervised(
        X_val, y_val, lookback=10
    )
    X_test, y_test = StockDataPreprocessor.time_series_to_supervised(
        X_test, y_test, lookback=10
    )

    # Initialize and run optimization
    rf_optimizer = RandomForestOptimizer()
    rf_optimizer.optimize(X_train, y_train, X_val, y_val)

    # Train final model with best parameters
    final_model = rf_optimizer.train_best_model(X_train, y_train)

    from model_evaluation import ModelEvaluator

    rf_evaluator = ModelEvaluator(
        model=final_model,
        model_type="RF",
    )

    metrics = rf_evaluator.evaluate(X_test, y_test)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    print("-" * 40)

    rf_evaluator.plot_confusion_matrix(X_test, y_test)

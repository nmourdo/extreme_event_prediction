import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
import numpy as np
import random

try:
    from src.data_preprocessing import StockDataPreprocessor
except ModuleNotFoundError:
    from data_preprocessing import StockDataPreprocessor


class RandomForestOptimizer:
    """A class for tuning RandomForest hyperparameters using Hyperopt.

    The class implements hyperparameter tuning for a RandomForestClassifier
    using the Tree Parzen Estimators (TPE) algorithm from Hyperopt. It optimizes
    for the F2 score, which puts more emphasis on recall over precision.

    Args:
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Attributes:
        random_state (int): Random seed for reproducibility.
        best_params (dict): Best hyperparameters found during optimization.
        best_model (RandomForestClassifier): Trained model with best parameters.
        trials (hyperopt.Trials): Results of all optimization trials.
        space (dict): Hyperparameter search space definition, including:
            - n_estimators (int): Number of trees [700-800]
            - max_depth (int): Maximum tree depth [70-100]
            - min_samples_split (int): Minimum samples for split [5-15]
            - min_samples_leaf (int): Minimum samples in leaf [1-3]
            - max_features (float): Maximum features ratio [0.8-1.0]
            - max_leaf_nodes (int): Maximum leaf nodes [500-600]
            - min_impurity_decrease (float): Minimum impurity decrease [0.04-0.05]

    Methods:
        optimize(X_train, y_train, X_val, y_val, max_evals=30):
            Performs hyperparameter optimization using validation data.

        train_best_model(X_train, y_train):
            Trains a new model using the best parameters found.

        save_model(filepath):
            Saves the best trained model to disk.

    Example:
        >>> optimizer = RandomForestOptimizer(random_state=42)
        >>> optimizer.optimize(X_train, y_train, X_val, y_val)
        >>> model = optimizer.train_best_model(X_train, y_train)
        >>> optimizer.save_model('best_model.pkl')
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.trials = None

        # Define hyperparameter search space
        self.space = {
            "n_estimators": hp.quniform("n_estimators", 700, 800, 1),
            "max_depth": hp.quniform("max_depth", 70, 100, 1),
            "min_samples_split": hp.quniform("min_samples_split", 5, 15, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 3, 1),
            "max_features": hp.uniform("max_features", 0.8, 1.0),
            "max_leaf_nodes": hp.quniform("max_leaf_nodes", 500, 600, 1),
            "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.04, 0.05),
        }

    def objective(self, params: dict) -> dict:
        """Objective function to be minimized by hyperopt.

        Trains a RandomForestClassifier with given parameters and returns
        the negative F2 score (for minimization).

        Args:
            params (dict): Hyperparameters to evaluate, including:
                - n_estimators: Number of trees
                - max_depth: Maximum tree depth
                - min_samples_split: Minimum samples for split
                - min_samples_leaf: Minimum samples in leaf
                - max_features: Maximum features ratio
                - max_leaf_nodes: Maximum leaf nodes
                - min_impurity_decrease: Minimum impurity decrease

        Returns:
            dict: Contains:
                - loss: Negative F2 score (to minimize)
                - status: Optimization status
        """
        # Get current hyperparameters
        model_params = self._process_params(params)
        # Add fixed parameters
        model_params.update(
            {
                "criterion": "entropy",
                "class_weight": "balanced",
                "bootstrap": True,
                "random_state": 42,
                "n_jobs": -1,
            }
        )
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(self.X_train, self.y_train)

        # Make predictions on the validation set
        y_pred = model.predict(self.X_val)
        f2 = fbeta_score(self.y_val, y_pred, beta=2)

        return {"loss": -f2, "status": STATUS_OK}

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_evals: int = 30,
    ) -> None:
        """Performs hyperparameter optimization using TPE algorithm.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            max_evals (int, optional): Maximum number of optimization iterations. Defaults to 30.

        Returns:
            dict: Best parameters found during optimization

        Prints:
            - Best F2 score achieved
            - Best parameters found
        """
        self.X_train = X_train
        self.y_train = y_train.astype(int)
        self.X_val = X_val
        self.y_val = y_val.astype(int)

        # Initialize trials object
        self.trials = Trials()

        # Perform optimization
        self.best_params = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
        )

        print("\nBest trial:")
        print(f"  Value (F2): {-min(self.trials.losses()):.4f}")
        print("\nBest parameters:")
        for key, value in self.best_params.items():
            print(f"    {key}: {value}")

        return self.best_params

    def _process_params(self, params: dict) -> dict:
        """Processes hyperparameters for model training.

        Converts numeric parameters to appropriate types and adds fixed parameters.

        Args:
            params (dict): Raw hyperparameters

        Returns:
            dict: Processed parameters ready for RandomForestClassifier
        """
        # Copy parameters to avoid modifying the original dictionary which is
        # used by hyperopt
        processed_params = params.copy()
        processed_params.update(
            {
                "n_estimators": int(processed_params["n_estimators"]),
                "max_depth": int(processed_params["max_depth"]),
                "min_samples_split": int(processed_params["min_samples_split"]),
                "min_samples_leaf": int(processed_params["min_samples_leaf"]),
                "max_leaf_nodes": int(processed_params["max_leaf_nodes"]),
                "criterion": "entropy",
                "class_weight": "balanced",
                "bootstrap": True,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        )

        return processed_params

    def train_best_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestClassifier:
        """Trains a new model using the best parameters found during optimization.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels

        Returns:
            RandomForestClassifier: Trained model with best parameters

        Raises:
            ValueError: If optimize() hasn't been called before training
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() before training the best model")

        best_params = self._process_params(self.best_params)
        self.best_model = RandomForestClassifier(**best_params)
        self.best_model.fit(X_train, y_train)
        return self.best_model

    def save_model(self, filepath: str) -> None:
        """Saves the trained model to disk using pickle.

        Args:
            filepath (str): Path where to save the model

        Raises:
            ValueError: If no model has been trained yet
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train the model first.")
        with open(filepath, "wb") as f:
            pickle.dump(self.best_model, f)


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

    # Convert time series data into supervised learning format (n_samples, n_features * n_lookback)
    X_train, y_train = preprocessor.time_series_to_supervised(
        X_train, y_train, lookback=10
    )
    X_val, y_val = preprocessor.time_series_to_supervised(X_val, y_val, lookback=10)
    X_test, y_test = preprocessor.time_series_to_supervised(X_test, y_test, lookback=10)

    # Initialize and run optimization
    print("-" * 50)
    print("Optimizing Random Forest hyperparameters...")
    print("-" * 50)
    rf_optimizer = RandomForestOptimizer()
    rf_optimizer.optimize(X_train.values, y_train.values, X_val.values, y_val.values)

    # Train final model with best parameters
    print("-" * 50)
    print("Training final model with best parameters...")
    print("-" * 50)
    final_model = rf_optimizer.train_best_model(X_train.values, y_train.values)

    # Save the model
    print("-" * 50)
    print("Saving final model...")
    print("-" * 50)
    rf_optimizer.save_model("best_random_forest.pkl")

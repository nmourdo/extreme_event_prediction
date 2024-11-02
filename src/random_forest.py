import pickle
from data_preprocessing import StockDataPreprocessor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score


class RandomForestOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = None
        self.best_model = None
        self.trials = None

        # Define hyperparameter search space
        self.class_weight_options = ["balanced", "balanced_subsample"]
        self.criterion_options = ["entropy", "gini"]
        self.space = {
            "n_estimators": hp.quniform("n_estimators", 600, 1000, 1),
            "max_depth": hp.quniform("max_depth", 70, 100, 1),
            "min_samples_split": hp.quniform("min_samples_split", 5, 15, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
            "max_features": hp.uniform("max_features", 0.8, 1.0),
            "max_leaf_nodes": hp.quniform("max_leaf_nodes", 400, 600, 1),
            "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.04, 0.08),
            "bootstrap": hp.choice("bootstrap", [True, False]),
            "class_weight": hp.choice("class_weight", self.class_weight_options),
            "criterion": hp.choice("criterion", self.criterion_options),
        }

    def objective(self, params):
        model_params = self._process_params(params, for_training=False)
        model = RandomForestClassifier(**model_params)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_val)
        f2 = fbeta_score(self.y_val, y_pred, beta=2)
        return {"loss": -f2, "status": STATUS_OK}

    def optimize(self, X_train, y_train, X_val, y_val, max_evals=30):
        self.X_train = X_train.values
        self.y_train = y_train.values.astype(int)
        self.X_val = X_val.values
        self.y_val = y_val.values.astype(int)

        self.trials = Trials()
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

    def _process_params(self, params, for_training=True):
        processed_params = params.copy()
        processed_params.update(
            {
                "n_estimators": int(processed_params["n_estimators"]),
                "max_depth": int(processed_params["max_depth"]),
                "min_samples_split": int(processed_params["min_samples_split"]),
                "min_samples_leaf": int(processed_params["min_samples_leaf"]),
                "max_leaf_nodes": int(processed_params["max_leaf_nodes"]),
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        )

        if for_training:
            processed_params["class_weight"] = self.class_weight_options[
                int(processed_params["class_weight"])
            ]
            processed_params["criterion"] = self.criterion_options[
                int(processed_params["criterion"])
            ]
            processed_params["bootstrap"] = processed_params["bootstrap"] == True

        return processed_params

    def train_best_model(self, X_train, y_train):
        if self.best_params is None:
            raise ValueError("Must run optimize() before training the best model")

        best_params = self._process_params(self.best_params, for_training=True)
        self.best_model = RandomForestClassifier(**best_params)
        self.best_model.fit(X_train, y_train)
        return self.best_model

    def save_model(self, filepath):
        if self.best_model is None:
            raise ValueError("No model to save. Train the model first.")
        with open(filepath, "wb") as f:
            pickle.dump(self.best_model, f)


if __name__ == "__main__":
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

    # Initialize and run optimization
    rf_optimizer = RandomForestOptimizer()
    rf_optimizer.optimize(X_train, y_train, X_val, y_val)

    # Train final model with best parameters
    final_model = rf_optimizer.train_best_model(X_train, y_train)

    # Save the model
    rf_optimizer.save_model("models/best_random_forest.pkl")

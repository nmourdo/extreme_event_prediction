import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class StockDataPreprocessor:
    """
    A class for preprocessing stock market data for machine learning applications.

    This class handles downloading, cleaning, and transforming stock data, with functionality
    for feature engineering and preparing data for time series analysis.

    Methods:
        download_and_prepare_stock_data(): Downloads and prepares initial stock data with extreme event labels
        standardize_data(): Standardizes features using StandardScaler from sklearn
        split_data(): Splits data into training, validation, and test sets
        time_series_to_supervised(): Transforms data into supervised learning format with lookback periods
        create_sequences(): Creates sequences for time series models
        add_features(): Adds technical indicators and market features

    Attributes:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data collection (YYYY-MM-DD format)
        end_date (str): End date for data collection (YYYY-MM-DD format)
        data (pd.DataFrame): Stored stock data after processing
    """

    def __init__(self, ticker: str, start_date: str, end_date: str) -> None:
        """
        Initialize the StockDataPreprocessor.

        Args:
            ticker (str): The stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_and_prepare_stock_data(self) -> pd.DataFrame:
        """
        Downloads stock data from Yahoo Finance for the specified ticker and date range,
        calculates daily returns, and creates a binary label for extreme price movements
        (defined as daily returns exceeding 2% in absolute value).

        Returns:
            pd.DataFrame: DataFrame containing the stock data with the following columns:
                - Standard OHLCV columns from Yahoo Finance
                - Daily_Returns: Percentage daily returns
                - Extreme_Event: Binary label indicating if next day has >2% price movement
        """
        # Download the stock data from Yahoo Finance
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Calculate the daily returns
        self.data["Daily_Returns"] = self.data["Adj Close"].pct_change() * 100
        # Remove the first row as it is a NaN value
        self.data = self.data.dropna()
        # Create the Extreme_Event column
        self.data["Extreme_Event"] = (abs(self.data["Daily_Returns"]) > 2).astype(int)
        # Shift the Extreme_Event column by one day
        self.data["Extreme_Event"] = self.data["Extreme_Event"].shift(-1)
        # Remove the last row as it is a NaN value
        self.data = self.data.dropna()
        return self.data

    @staticmethod
    def standardize_data(
        data: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """
        Standardize the data for the given feature columns. Each feature from the transformed
        data will have a mean of 0 and a standard deviation of 1.

        Args:
            data (pd.DataFrame): Input DataFrame
            feature_columns (list[str]): List of column names to standardize

        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        scaler = StandardScaler()
        df = data.copy()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        return df

    @staticmethod
    def split_data(
        data: pd.DataFrame,
        feature_columns: list[str],
        label: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.85,
    ) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
    ]:
        """
        Splits the data into training, validation and testing sets. As this is time series data,
        the splits are made chronologically without shuffling: the first portion of the data is used
        for training, the middle portion for validation, and the final portion for testing.

        Args:
            data (pd.DataFrame): Input DataFrame containing features and labels
            feature_columns (list[str]): List of column names to use as features
            label (str): Column name to use as label
            train_ratio (float): Proportion of data to use for training (default: 0.7)
            val_ratio (float): Total proportion for training + validation (default: 0.85).
                              The validation set will use (val_ratio - train_ratio) of the data,
                              and the remaining (1 - val_ratio) will be used for testing.

        Returns:
            Tuple containing:
                X_train (pd.DataFrame): Training features (earliest time period)
                y_train (pd.Series): Training labels
                X_val (pd.DataFrame): Validation features (middle time period)
                y_val (pd.Series): Validation labels
                X_test (pd.DataFrame): Test features (most recent time period)
                y_test (pd.Series): Test labels

        Raises:
            ValueError: If column names are not found in the DataFrame
        """
        # Validate inputs
        missing_features = [col for col in feature_columns if col not in data.columns]

        if missing_features:
            raise ValueError(
                f"Feature columns not found in DataFrame: {missing_features}"
            )
        if label not in data.columns:
            raise ValueError(f"Label column not found in DataFrame: {label}")

        # Extract features and labels
        features = data[feature_columns]
        labels = data[label].astype(int)

        # Calculate split points
        train_size = int(train_ratio * len(data))
        val_size = int(val_ratio * len(data))

        # Split into train, validation and test sets
        X_train = features.iloc[:train_size]
        y_train = labels.iloc[:train_size]

        X_val = features.iloc[train_size:val_size]
        y_val = labels.iloc[train_size:val_size]

        X_test = features.iloc[val_size:]
        y_test = labels.iloc[val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def time_series_to_supervised(
        X: pd.DataFrame, y: pd.Series, lookback: int = 10, dropnan: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare time series data for classification by creating sequences of historical features
        and their corresponding future labels.

        The shape of the output dataframe will be (n_samples - lookback, n_features * lookback).

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame containing feature columns
        y : pd.Series
            Series containing target labels
        lookback : int, default=10
            Number of historical time steps to use for prediction
        dropnan : bool, default=True
            Whether to remove rows with NaN values

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            X: DataFrame with historical features
            y: Series with target labels
        """
        # Clean column names - remove the ticker part
        X.columns = [col[0] if isinstance(col, tuple) else col for col in X.columns]

        cols = []
        names = []

        # Create lagged feature columns
        for i in range(lookback, 0, -1):
            cols.append(
                X.shift(i)
            )  # Shifts entire DataFrame, creating all lagged features
            for col in X.columns:  # Creates corresponding names for all shifted columns
                names.append(f"{col}(t-{i})")

        # Combine all lagged features
        X_transformed = pd.concat(cols, axis=1)
        X_transformed.columns = names

        # Prepare labels (predict next day)
        y_transformed = y[lookback:]

        # Remove rows with NaN if specified
        if dropnan:
            X_transformed = X_transformed.dropna()
            y_transformed = y_transformed[: len(X_transformed)]

        assert len(X_transformed) == len(
            y_transformed
        ), "Features and labels must have same length"

        return X_transformed, y_transformed

    @staticmethod
    def create_sequences(
        X: pd.DataFrame, y: pd.Series, lookback: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences of lookback days for each feature, with corresponding labels. This method transforms the data
        into a proper format for sequential deep learning models.The output shape will be (n_samples - lookback, n_features, lookback).

        Parameters:
        -----------
        X : pd.DataFrame
            Input features dataframe with datetime index
        y : pd.Series
            Input labels series with datetime index
        lookback : int
            Number of timesteps to look back (default: 10)

        Returns:
        --------
        X_seq : numpy.ndarray
            Array of shape [n_samples, n_features, lookback]
        y_seq : numpy.ndarray
            Array of labels
        """
        n_features = X.shape[1]
        n_samples = len(X) - lookback

        # Initialize arrays
        X_seq = np.zeros((n_samples, n_features, lookback))
        y_seq = np.zeros(n_samples)

        # Create sequences
        for i in range(n_samples):
            # Get sequence of lookback days for each feature
            sequence = X.iloc[i : i + lookback]
            # Store features (reverse order so most recent is last)
            X_seq[i] = sequence.T.values[:, ::-1]
            # Store label (the extreme event flag for the next day after sequence)
            y_seq[i] = y.iloc[i + lookback]

        return X_seq, y_seq

    def add_features(self) -> pd.DataFrame:
        """
        Add new features to the existing DataFrame, dropping NaN values after each feature
        calculation to ensure data quality and traceability.

        Features added:
            - rolling_volatility: 10-day rolling standard deviation of returns
            - relative_volume: Current volume relative to 10-day moving average
            - VIX: Market volatility index
            - bollinger_band_width: Width of Bollinger Bands (2 standard deviations)
            - ATR: 10-day Average True Range

        Returns:
            pd.DataFrame: DataFrame with added technical indicators and market features
        """
        # 1. 10-day Rolling Volatility
        returns = self.data["Close"].pct_change()
        self.data["rolling_volatility"] = returns.rolling(window=10).std()
        self.data = self.data.dropna()

        # 2. Volume Relative to 10-day MA
        volume_ma = self.data["Volume"].rolling(window=10).mean()
        self.data["relative_volume"] = self.data["Volume"] / volume_ma
        self.data = self.data.dropna()

        # 3. Download VIX and add directly
        vix = yf.download("^VIX", start=self.start_date, end=self.end_date)["Close"]
        self.data["VIX"] = vix
        self.data = self.data.dropna()

        # 4. Bollinger Band Width
        sma = self.data["Close"].rolling(window=10).mean()
        std = self.data["Close"].rolling(window=10).std()
        self.data["bollinger_band_width"] = (2 * std) / sma  # Normalized by price level
        self.data = self.data.dropna()

        # 5. Average True Range (ATR)
        high_low = self.data["High"] - self.data["Low"]
        high_close = abs(self.data["High"] - self.data["Close"].shift())
        low_close = abs(self.data["Low"] - self.data["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.data["ATR"] = true_range.rolling(window=10).mean()
        self.data = self.data.dropna()

        return self.data

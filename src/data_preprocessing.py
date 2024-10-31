import yfinance as yf
import pandas as pd
from typing import Tuple


def download_and_prepare_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock data and prepares it for extreme event prediction.
    
    Downloads stock data from Yahoo Finance for the specified ticker and date range,
    calculates daily returns, and creates a binary label for extreme price movements
    (defined as daily returns exceeding 2% in absolute value).

    Args:
        ticker (str): The stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame containing the stock data with the following columns:
            - Standard OHLCV columns from Yahoo Finance
            - Daily_Returns: Percentage daily returns
            - Extreme_Event: Binary label indicating if next day has >2% price movement
    """
    # Download the stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # Calculate the daily returns
    stock_data['Daily_Returns'] = stock_data['Adj Close'].pct_change() * 100
    # Remove the first row as it is a NaN value
    stock_data = stock_data.dropna()
    # Create the Extreme_Event column
    stock_data['Extreme_Event'] = (abs(stock_data['Daily_Returns']) > 2).astype(int)
    # Shift the Extreme_Event column by one day
    stock_data['Extreme_Event'] = stock_data['Extreme_Event'].shift(-1)
    # Remove the last row as it is a NaN value
    stock_data = stock_data.dropna()
    return stock_data

def split_data(data: pd.DataFrame, train_ratio: float = 0.8, val_ratio : float = 0.2, feature_end_idx: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training, validation and testing sets.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing features and labels
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        val_ratio (float): Ratio of data to use for validation (default: 0.2)
        feature_end_idx (int): Index up to which columns should be considered as features (default: 7)
                              First feature_end_idx columns will be used as features
    
    Returns:
        Tuple containing:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_val (pd.DataFrame): Validation features  
            y_val (pd.Series): Validation labels
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
    """
    # Extract features and labels
    features = data.iloc[:, :feature_end_idx]  
    labels = data.iloc[:, -1]    # Last column (Extreme_Event_Tomorrow)

    # Define train/val/test split ratios
    train_ratio = 0.8
    val_ratio = 0.9

    # Split into train, validation and test sets
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))

    X_train = features.iloc[:train_size]
    y_train = labels.iloc[:train_size]

    X_val = features.iloc[train_size:val_size]
    y_val = labels.iloc[train_size:val_size]

    X_test = features.iloc[val_size:]
    y_test = labels.iloc[val_size:]
            
    return X_train, y_train, X_val, y_val, X_test, y_test


def time_series_to_supervised(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int = 10,
    dropnan: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare time series data for classification by creating sequences of historical features
    and their corresponding future labels.
    
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
        cols.append(X.shift(i))        # Shifts entire DataFrame, creating all lagged features
        for col in X.columns:          # Creates corresponding names for all shifted columns
            names.append(f'{col}(t-{i})')
    
    # Combine all lagged features
    X_transformed = pd.concat(cols, axis=1)
    X_transformed.columns = names
    
    # Prepare labels (predict next day)
    y_transformed = y[lookback:]
    
    # Remove rows with NaN if specified
    if dropnan:
        X_transformed = X_transformed.dropna()
        y_transformed = y_transformed[:len(X_transformed)]
    
    assert len(X_transformed) == len(y_transformed), "Features and labels must have same length"
    
    return X_transformed, y_transformed
def add_features(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Add new features to the existing DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns ['Close', 'Volume']
    start_date : str
        Start date for VIX data in format 'YYYY-MM-DD'
    end_date : str
        End date for VIX data in format 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added features: rolling_volatility, relative_volume, and VIX
    """
    # 1. 10-day Rolling Volatility
    returns = df['Close'].pct_change()
    df['rolling_volatility'] = returns.rolling(window=10).std()
    
    # 2. Volume Relative to 10-day MA
    volume_ma = df['Volume'].rolling(window=10).mean()
    df['relative_volume'] = df['Volume'] / volume_ma
    
    # 3. Download VIX and add directly
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    df['VIX'] = vix
    
    # Drop rows with NaN values from the rolling calculations
    df = df.dropna()
    
    return df
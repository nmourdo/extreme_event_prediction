# Stock Market Extreme Event Prediction

This project implements machine learning models to predict extreme events (>2% daily price movements) in Apple stock prices using historical data. It compares Random Forest, Temporal CNN, and LSTM approaches.

## Project Structure

```
.
├── data/
│   ├── models/          # Saved model checkpoints 
│   └── figures/         # Generated plots and visualizations
├── src/
│   ├── data_preprocessing.py  # Data preparation utilities
│   ├── random_forest.py       # Random Forest implementation
│   ├── temporal_cnn.py        # TCNN implementation
│   ├── improvement.py         # LSTM and additional models
│   ├── model_evaluation.py    # Evaluation metrics and visualization
│   └── report.ipynb          # Analysis and results notebook
├── README.md
├── pyproject.toml      # Poetry dependencies
└── poetry.lock        # Poetry lock file
```

## Setup Instructions

1. Install Poetry for dependency management:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction
poetry install
```

3. Create required directories:
```bash
mkdir -p data/models data/figures
```

## Running the Code

Execute the scripts in the following order:

1. Data preprocessing and Random Forest training:
```bash
poetry run python src/random_forest.py
```

2. Train the TCNN model:
```bash
poetry run python src/temporal_cnn.py
```

3. Train the LSTM model:
```bash
poetry run python src/improvement.py
```

4. Evaluate and compare models:
```bash
poetry run python src/model_evaluation.py
```

5. View the full analysis:
```bash
poetry run jupyter notebook src/report.ipynb
```

## Dependencies

Key dependencies (managed by Poetry):
- Python 3.10+
- PyTorch
- scikit-learn
- pandas
- numpy
- yfinance
- matplotlib
- seaborn
- jupyter

See `pyproject.toml` for complete list.

## Data

The project uses Apple (AAPL) stock data from Yahoo Finance from January 2015 to January 2024. Data is downloaded automatically when running the scripts.

## Model Details

- **Random Forest**: Optimized using Hyperopt with F2 score objective
- **TCNN**: Temporal Convolutional Network with 2 conv layers
- **LSTM**: Long Short-Term Memory network with additional technical indicators

## Results

Model performance metrics and visualizations are saved to:
- `data/figures/` - Confusion matrices and prediction plots
- `data/models/` - Best model checkpoints

See `src/report.ipynb` for detailed analysis.

## Reproducibility

Random seeds are set in each script for reproducibility. Key parameters:
- Training/Validation/Test split: 70%/15%/15%
- Lookback period: 10 days
- Extreme event threshold: 2% price movement

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
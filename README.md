# Stock Market Extreme Event Prediction

This project implements machine learning models to predict extreme events (>2% daily price movements) in Apple stock prices using historical data. It compares Random Forest, Temporal CNN, and LSTM approaches.

## Project Structure

```
.
├── data/
│   ├── models/          # Saved models
│   └── figures/         # Some generated plots used in the report
├── src/
│   ├── data_preprocessing.py  # Data preparation utilities
│   ├── random_forest.py       # Random Forest implementation
│   ├── temporal_cnn.py        # TCNN implementation
│   ├── model_evaluation.py    # Evaluation of the random forest and the TCNN models specifified in the assigment description
│   ├── improvement.py         # RF, LSTM and MLP implementations with additional features
│   └── report.ipynb          # Analysis and results notebook
├── report.pdf            # PDF version of the report notebook
├── README.md
├── pyproject.toml      # Poetry dependencies
├── poetry.lock        # Poetry lock file
└── .gitignore        # Git ignore file
```

## Setup Instructions

1. Install Poetry for dependency management:
   Follow the installation instructions at https://python-poetry.org/docs/

2. Unzip the project, navigate to the project directory and install dependencies using Poetry:
```bash
poetry env use python3.10
poetry install
```

## Running the Code

The report has been included in the repository as a PDF file and as a Jupyter notebook. Feel free to run the notebook if you wish to see the results in an interactive way. For doing so, be sure to choose the appropriate kernel in the top right corner of the notebook, which will be created automatically when running the `poetry install` command.

As the data format expected by the different models is not the same, the data is generated each time the models are trained. Run the scripts in the following order:

1. Random Forest training:
```bash
poetry run python src/random_forest.py
```

2. Train the TCNN model:
```bash
poetry run python src/temporal_cnn.py
```

3. Evaluate and compare models the random forest and the TCNN models:
```bash
poetry run python src/model_evaluation.py
```

4. Train the random forest, MLP and LSTM models with extra features and evaluate them:
```bash
poetry run python src/improvement.py
```


## Documentation

The code has been thoroughly documented with docstrings, type hints and comments on multiple levels (module, class and function).

## Dependencies

Key dependencies (managed by Poetry):
- Python 3.10+
- PyTorch
- scikit-learn
- Lightning
- Hyperopt
- Ray Tune
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

See `src/report.ipynb` for detailed analysis.

## Reproducibility

Random seeds are set in each script for reproducibility. Key parameters:

It is noted that the best neural network models (TCNN and LSTM) have been saved in the repository and are used in the evaluation scripts and the report notebook. This is because even though all seeds are set, they were not retrieved across multiple runs.
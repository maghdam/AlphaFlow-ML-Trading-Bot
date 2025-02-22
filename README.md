# AlphaFlow ML Trading Bot Project

A comprehensive **machine learning trading framework** that covers the entire workflow:

1. **Data loading** from MetaTrader 5
2. **Feature engineering** (technical indicators, custom features, labeling)
3. **Model training** (RandomForest, XGBoost, LightGBM, etc.)
4. **Hyperparameter tuning** (RandomizedSearchCV or GridSearchCV)
5. **Time-based / walk-forward cross-validation**
6. **Backtesting** (VectorBT or simple custom code)
7. **Live trading** integration with MetaTrader 5

### Supported Labeling Strategies:
- **Regression** on next-bar returns
- **Multi-bar classification**
- **Double-barrier labeling** (López de Prado style)
- **Regime detection** (simple up/down/sideways approach)

This project provides a flexible **template** for you to **create and add your own** custom labeling functions or feature engineering steps, allowing you to experiment with new ideas and strategies.

## Table of Contents
1. [Features](#features)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Usage](#usage)
    - Backtesting Notebooks
    - Live Trading Scripts
5. [Key Modules](#key-modules)
6. [Extending the Project](#extending-the-project)
7. [Disclaimer](#disclaimer)
8. [License](#license)

## Features
- **MetaTrader 5** data retrieval (`data_loader.py`)
- **TA** library for feature engineering (`ta.add_all_ta_features`)
- Multiple **labeling methods**: multi-bar, double-barrier, regime detection, etc.
- **Time-based** or **walk-forward** cross-validation to avoid data leakage
- **RandomizedSearchCV** or **GridSearchCV** for hyperparameter tuning
- **VectorBT** or custom backtesting scripts for performance evaluation
- **Live trading** scripts with real-time MetaTrader 5 order sending

## Repository Structure
```bash
ml_bot_trading/
├── data/
│   ├── data_loader.py  # MetaTrader 5 data retrieval
│
├── features/
│   ├── feature_engineering.py  # Technical indicators, custom features
│   ├── labeling.py  # Labeling methods: multi-bar, double-barrier, regime detection
│
├── models/
│   ├── model_training.py  # Model selection, hyperparam tuning
│   ├── saved_models/  # Folder for .pkl pipelines (best_rf_pipeline.pkl, etc.)
│
├── backtests/
│   ├── simple_backtest.py  # Simple Pythonic backtest logic
│   ├── vectorbt_backtest.py  # VectorBT-based backtesting template
│
├── live_trading/
│   ├── regression_returns.py  # Live trading script for regression returns
│   ├── multi_bar.py  # Live trading script for multi-bar classification
│   ├── double_barrier.py  # Live trading script for double-barrier labeling
│   ├── regime_detection.py  # Live trading script for regime detection
│
├── notebooks/
│   ├── 01_backtests_regression_returns.ipynb
│   ├── 01_live_trading_regression_returns.ipynb
│   ├── 02_backtests_multi_bar_classification.ipynb
│   ├── 02_live_trading_multi_bar.ipynb
│   ├── 03_backtests_double_barrier.ipynb
│   ├── 03_live_trading_double_barrier.ipynb
│   ├── 04_backtests_regime_detection.ipynb
│   ├── 04_live_trading_regime_detection.ipynb
│
├── requirements.txt
├── README.md
```

## Setup & Installation

### 1. Clone this repository:
```bash
git clone https://github.com/YourUsername/ml_bot_trading.git
cd ml_bot_trading
```

### 2. Create and activate a Python environment (conda or venv):
```bash
conda create -n ml_trading python=3.9
conda activate ml_trading
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
- Make sure you have **MetaTrader5** installed if you plan to do live trading.

### 4. (Optional) Install Jupyter Notebook:
```bash
pip install jupyter
```

## Usage
### Backtesting Notebooks
1. Navigate to `notebooks/`, pick a relevant file (e.g., `02_backtests_multi_bar_classification.ipynb`), and run it:
   ```bash
   jupyter notebook
   ```
2. Inside the notebook, you can see how we do:
   - Feature engineering
   - Labeling
   - Walk-forward splits
   - Train & tune
   - VectorBT or custom backtesting

### Live Trading Scripts
1. Go to `live_trading/` folder and pick the script for your labeling approach:
   - `multi_bar.py`
   - `double_barrier.py`
   - `regime_detection.py`
2. Adjust **MetaTrader 5 credentials** (login, server, password) in the script.
3. Run from terminal:
   ```bash
   python live_trading/double_barrier.py
   ```
4. The script will:
   - Load the pipeline (e.g., `final_production_pipeline.pkl`)
   - Fetch new bars from MetaTrader 5
   - Predict SHIFTED classes `[0, 1, 2]` => SHIFT back to `[-1, 0, +1]`
   - Place orders if signals = ±1

## Key Modules
- **`data/data_loader.py`**: Connects to MetaTrader 5, fetches bars with `copy_rates_from_pos`.
- **`features/feature_engineering.py`**: Uses the **TA** library and additional custom features (spreads, autocorrelation, etc.).
- **`features/labeling.py`**:
  - `create_labels_multi_bar(...)`
  - `create_labels_double_barrier(...)`
  - `create_labels_regime_detection(...)`
- **`models/model_training.py`**:
  - `select_features_rf_reg(...)`
  - Time-based splits, random/grid search for hyperparams.
- **`backtests/`**:
  - `simple_backtest.py` or `vectorbt_backtest.py`
- **`live_trading/`**:
  - Each script loads a pipeline (`.pkl`), connects to MT5, and places trades based on predictions.

## Extending the Project
- **Add your own label**: Create a new function in `features/labeling.py` (e.g. `create_labels_custom(...)` that returns a new column with `[-1, 0, +1]` (or your custom classes)).
- **Add your own features**: Implement them in `features/feature_engineering.py` or create a new file.
- **Train a new model**: Adapt `models/model_training.py` or your notebooks to handle new classifiers/regressors.
- **Explore new backtest approaches**: Either integrate with `vectorbt` in a notebook or write a custom `.py` in `backtests/`.

## Disclaimer
We share this code for **learning and development/research purposes only**. Nothing herein constitutes financial advice or a recommendation to trade real money. **Trading involves substantial risk.** Always do your own due diligence, consult professionals, and only risk capital you can afford to lose.

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

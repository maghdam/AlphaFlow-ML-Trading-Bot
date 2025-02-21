# model_training.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Splits df into train and test sets chronologically.
    """
    split_index = int(len(df) * train_ratio)
    df_train = df.iloc[:split_index].copy()
    df_test = df.iloc[split_index:].copy()
    return df_train, df_test

def select_features_rf_reg(X, y, estimator=None, max_features=20):
    """
    Uses a random forest (or a user-provided estimator) to rank feature importances,
    then keeps the top 'max_features'.
    Returns (X_new, selected_features_indices).
    """
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)

    estimator.fit(X, y)
    importances = estimator.feature_importances_
    # Sort by importance descending
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:max_features]
    X_new = X.iloc[:, top_indices]
    return X_new, top_indices


def train_random_forest_reg(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a RandomForestRegressor and returns the fitted model.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def evaluate_regression(model, X_test, y_test) -> dict:
    """
    Evaluates a regression model with MSE and MAE.
    Returns a dict with {'mse': ..., 'mae': ...}.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return {"mse": mse, "mae": mae}



def train_and_evaluate_reg_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Trains and evaluates each model in 'models' dict.
    Returns a dict of MSE results for each model.
    """
    from sklearn.metrics import mean_squared_error

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[model_name] = mse
    return results

def train_and_evaluate_reg_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Trains and evaluates each model in 'models' dict.
    Returns a dict of MSE results for each model.
    """
    from sklearn.metrics import mean_squared_error
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[model_name] = mse
    return results

def walk_forward_splits(X, y, n_splits=3):
    """
    Splits the data X, y into multiple chronological folds.
    For example, with n_splits=3, we do:
      - Fold 1: Train [0 : fold1], Test [fold1 : fold2]
      - Fold 2: Train [0 : fold2], Test [fold2 : fold3]
      - Fold 3: Train [0 : fold3], Test [fold3 : end]

    The size of each test fold is len(X) // (n_splits + 1).

    Returns a list of (X_train, y_train, X_test, y_test) tuples.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    folds = []

    for i in range(n_splits):
        start_test = (i + 1) * fold_size
        end_test = (i + 2) * fold_size
        if end_test > n:
            end_test = n

        # Train = [0 : start_test]
        X_train_fold = X.iloc[:start_test]
        y_train_fold = y.iloc[:start_test]

        # Test = [start_test : end_test]
        X_test_fold = X.iloc[start_test:end_test]
        y_test_fold = y.iloc[start_test:end_test]

        folds.append((X_train_fold, y_train_fold, X_test_fold, y_test_fold))

    return folds



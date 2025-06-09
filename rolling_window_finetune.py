from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score

from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# RandomizedSearchCV for RF
def rolling_window_validation_rf(params, X, y, train_size, test_size, step_size):
    mse_scores = []
    n_samples = len(X)
    
    # Perform rolling window
    for start in range(0, n_samples - train_size - test_size + 1, step_size):
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = train_end + test_size

        # Split the data
        X_train, X_test = X[train_start:train_end], X[test_start:test_end]
        y_train, y_test = y[train_start:train_end], y[test_start:test_end]

        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))

    return np.mean(mse_scores)


import optuna
import warnings
warnings.filterwarnings('ignore')

# Rolling window validation function for LightGBM
def rolling_window_validation_lgb(params, X, y, train_size, test_size, step_size):
    mse_scores = []
    n_samples = len(X)
    
    # Perform rolling window
    for start in range(0, n_samples - train_size - test_size + 1, step_size):
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = train_end + test_size

        # Split the data
        X_train, X_test = X[train_start:train_end], X[test_start:test_end]
        y_train, y_test = y[train_start:train_end], y[test_start:test_end]

        # Train the model
        model = lgb.LGBMRegressor(**params, random_state=42)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",  # Use RMSE as the evaluation metric
        )

        # Predict and calculate MSE
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        mse_scores.append(mean_squared_error(y_test, y_pred))

    return np.mean(mse_scores)

def rolling_window_validation_xgb(params, X, y, train_size, test_size, step_size):
    mse_scores = []
    n_samples = len(X)

    # Perform rolling window
    for start in range(0, n_samples - train_size - test_size + 1, step_size):
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = train_end + test_size

        # Split the data
        X_train, X_test = X[train_start:train_end], X[test_start:test_end]
        y_train, y_test = y[train_start:train_end], y[test_start:test_end]

        # Train the model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dtest, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # Predict without specifying ntree_limit
        y_pred = model.predict(dtest)
        mse_scores.append(mean_squared_error(y_test, y_pred))

    return np.mean(mse_scores)


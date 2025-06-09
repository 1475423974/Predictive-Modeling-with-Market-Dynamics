import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score


from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def evaluate_model_performance(y_test, y_pred, modelname, train_set = False, rolling_window=30):
    """
    Evaluate model performance with metrics including MSE, MAE, R-squared, Rolling Window MSE, 
    Directional Accuracy, and Explained Variance.

    Returns:
    - metrics: dict, containing evaluation metrics
    """
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate R-squared (R²)
    r_squared = r2_score(y_test, y_pred)
    
    # Calculate Explained Variance Score
    explained_variance = explained_variance_score(y_test, y_pred)
    
    # Calculate Rolling Window MSE
    rolling_mse = [
        mean_squared_error(y_test[i:i+rolling_window], y_pred[i:i+rolling_window])
        for i in range(len(y_test) - rolling_window + 1)
    ]
    rolling_mse_avg = np.mean(rolling_mse) if rolling_mse else None  # Handle edge cases
    
    # Calculate Directional Accuracy
    direction_actual = np.sign(np.diff(y_test))  # Actual direction (up/down)
    direction_pred = np.sign(np.diff(y_pred))   # Predicted direction (up/down)
    
    # Ensure lengths match
    if len(direction_actual) == len(direction_pred):
        directional_accuracy = accuracy_score(direction_actual, direction_pred)
    else:
        directional_accuracy = None  
    
    metrics = {
        "MSE": round(mse,2),
        "MAE": round(mae,2),
        "Rolling Window MSE": round(rolling_mse_avg,2),
        "R-squared (R²)": round(r_squared,2),
        "Explained Variance": round(explained_variance,2),
        "Directional Accuracy (%)": round(directional_accuracy*100,2)
    }
    if train_set == True:
        train_set = "train"
    if train_set == False:
        train_set = "test"
    print(f"Performance Metrics for {modelname} ({train_set})")
    return pd.DataFrame(metrics, index=[f'{modelname}_{train_set}']).T


def plot_fitting(y, y_model, y_model_finetuned, modelname, split_date = '2019-06-20'):
    y_model = pd.Series(y_model)
    y_model_finetuned = pd.Series(y_model_finetuned)
    y_model.index = y.index
    y_model_finetuned.index = y.index
    plt.figure(figsize=(12, 6))
    

    plt.plot( y, label='Actual (y_test)', color='blue', linewidth=2)
    plt.plot( y_model, label=f'Prediction ({modelname})', color='orange', linestyle='--', linewidth=2)
    # plt.plot( y_model_finetuned, label=f'Prediction ({modelname} Fine-tuned)', color='green', linestyle='-.', linewidth=2)
    
    # Add vertical line at the split date
    if split_date!= None:
        plt.axvline(pd.to_datetime(split_date), color='red', linestyle='--', linewidth=1.5, label='Train-Test Split')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45) 
    
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Comparison of Actual and Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# y_lgb = lgb_model.predict(X)
# y_lgb_finetuned = lgb_model_finetuned.predict(X)
# plot_fitting(y, y_lgb, y_lgb_finetuned, 'LGB')
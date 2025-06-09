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

def backtest_strategy_cta(df, initial_capital, stop_loss_trailing, take_profit, holding_period, rolling_window=10):
    """
    Backtest a CTA-style strategy using rolling predictions to generate signals.
    Each trade uses the entire available capital. Includes a benchmark for comparison.

    Parameters:
        df (pd.DataFrame): DataFrame containing historical data and predictions.
            Must include:
                - 'LL100': Asset price (e.g., closing price).
                - 'LL100_fpred': Model-generated predictions.
        initial_capital (float): Initial capital for the strategy.
        stop_loss_trailing (float): Trailing stop-loss percentage.
        take_profit (float): Take-profit percentage.
        holding_period (int): Maximum holding period in days.
        rolling_window (int): Rolling window size for generating signals.

    Returns:
        pd.DataFrame: DataFrame containing trade details with benchmark capital.
    """
    df = df.copy()
    
    # Initialize columns
    df['Signal'] = 0  
    df['PnL'] = 0  
    
    # Generate Momentum signals 
    df['Rolling_Prediction'] = df['LL100_fpred'].rolling(rolling_window).mean()
    df['Signal'] = df['Rolling_Prediction'].diff()  
    df['Signal'] = df['Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    trades = []  # List to store trade details
    capital = initial_capital  # Starting capital
    position_active = False  # Whether a position is currently active
    position_end_date = None  # End date for the current position
    
    # Benchmark: Buy-and-hold strategy
    benchmark_shares = initial_capital / df['LL100'].iloc[0]  
    
    for i in range(len(df)):
        # If a position is active, check for exit conditions
        if position_active:
            exit_price = df['LL100'].iloc[i]  # Current price
            pnl = 0  
            
            # Update peak price (used for trailing stop-loss)
            if entry_signal == 1:  # Long position
                peak_price = max(peak_price, exit_price)
            elif entry_signal == -1:  # Short position
                peak_price = min(peak_price, exit_price)
            
            # Check trailing stop-loss condition
            if entry_signal == 1 and (exit_price - peak_price) / peak_price <= -stop_loss_trailing:
                pnl = (exit_price - entry_price) * position_size  # Long PnL
                position_active = False  # Exit position
            elif entry_signal == -1 and (peak_price - exit_price) / peak_price <= -stop_loss_trailing:
                pnl = (entry_price - exit_price) * position_size  # Short PnL
                position_active = False  # Exit position
            
            # Check take-profit condition
            if entry_signal == 1 and (exit_price - entry_price) / entry_price >= take_profit:
                pnl = (exit_price - entry_price) * position_size  
                position_active = False  
            elif entry_signal == -1 and (entry_price - exit_price) / entry_price >= take_profit:
                pnl = (entry_price - exit_price) * position_size 
                position_active = False 
            
            # Check holding period expiration condition
            if df.index[i] >= position_end_date:
                if entry_signal == 1:
                    pnl = (exit_price - entry_price) * position_size  
                elif entry_signal == -1:
                    pnl = (entry_price - exit_price) * position_size  
                position_active = False 
            
            # If the position is exited, record the trade
            if not position_active:
                capital += pnl  # Update capital with the PnL of the trade
                benchmark_capital = benchmark_shares * df['LL100'].iloc[i]  
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': df.index[i],
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Position Size': position_size,  
                    'PnL': pnl,
                    'Remaining Capital': capital,  # updated capital after the trade
                    'Benchmark Capital': benchmark_capital  # benchmark capital
                })
        
        # If no position is active, check for entry conditions
        if not position_active and df['Signal'].iloc[i] != 0:
            entry_price = df['LL100'].iloc[i]  
            entry_signal = df['Signal'].iloc[i] 
            position_size = capital / entry_price  
            peak_price = entry_price  
            entry_date = df.index[i] 
            position_active = True 
            position_end_date = df.index[min(i + holding_period, len(df) - 1)]  
    
    trades_df = pd.DataFrame(trades)
    
    return trades_df, df

def plot_buy_sell_signals(df_new):
    df_new['Buy_Signal'] = df_new['Signal'] == 1  # Long signals
    df_new['Sell_Signal'] = df_new['Signal'] == -1  # Short signals

    # Plot the price and signals
    plt.figure(figsize=(10, 6))
    plt.plot(df_new.index, df_new['LL100'], label="Price (LL100)", color="blue", alpha=0.7)
    plt.scatter(df_new.index[df_new['Buy_Signal']], df_new['LL100'][df_new['Buy_Signal']], label="Buy Signal", marker="^", color="green", s=20)
    plt.scatter(df_new.index[df_new['Sell_Signal']], df_new['LL100'][df_new['Sell_Signal']], label="Sell Signal", marker="v", color="red", s=20) 

    plt.title("Trading Signals Generated by the Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price (LL100)")
    plt.legend()
    plt.grid()
    plt.show()



def calculate_performance_metrics(trades_df, df, fee_rate=0.005, start_date=None, end_date=None):
    """
    Calculate the performance metrics for CTA and benchmark strategies, including:
    1. Annualized return
    2. Sharpe ratio
    3. Maximum drawdown
    4. Net returns considering transaction fees

    Parameters:
    - trades_df: A DataFrame containing trade records, which must include the following columns:
        - 'Entry Date': The start date of the trade
        - 'Exit Date': The end date of the trade
        - 'Remaining Capital': The remaining capital for the CTA strategy
        - 'Benchmark Capital': The capital for the benchmark strategy
    - fee_rate: The transaction fee rate for each trade, default is 0.005 (0.5%)
    - start_date: The start date of the backtesting period (string or datetime, e.g., '2017-01-01')
    - end_date: The end date of the backtesting period (string or datetime, e.g., '2020-12-31')

    Returns:
    - performance_metrics: A dictionary containing the performance metrics for CTA and benchmark strategies
    - daily_data: A DataFrame containing daily returns and capital data
    """

    trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
    trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date'])
    df = df.loc[start_date:end_date, ['LL100']]

    if start_date is None:
        start_date = trades_df['Entry Date'].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = trades_df['Exit Date'].max()
    else:
        end_date = pd.to_datetime(end_date)


    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')


    daily_data = pd.DataFrame({'Date': full_date_range})
    daily_data.set_index('Date', inplace=True)

    trades_df = trades_df.sort_values('Entry Date').reset_index(drop=True)

    daily_data['Remaining Capital'] = np.nan
    daily_data['Benchmark Capital'] = np.nan
    df['LL100_pct_change'] = df['LL100'].pct_change().fillna(0)
    for _, row in trades_df.iterrows():
        daily_data.loc[row['Exit Date'], 'Remaining Capital'] = row['Remaining Capital']
        daily_data.loc[row['Exit Date'], 'Benchmark Capital'] = df.loc[row['Exit Date'], 'LL100'] * (1+df.loc[row['Exit Date'], 'LL100_pct_change'])

    daily_data.fillna(method='ffill', inplace=True)
    daily_data.fillna(0, inplace=True)


    daily_data['CTA Daily Return'] = daily_data['Remaining Capital'].pct_change().fillna(0)
    daily_data['Benchmark Daily Return'] = daily_data['Benchmark Capital'].pct_change().fillna(0)
    daily_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    daily_data.fillna(0, inplace=True)

    def annualized_return(daily_return, periods_per_year=365):
        return (1 + daily_return.mean()) ** periods_per_year - 1

    def sharpe_ratio(daily_return, risk_free_rate=0.02, periods_per_year=365):
        excess_return = daily_return - (risk_free_rate / periods_per_year)
        if excess_return.std() == 0:
            return 0
        return np.sqrt(periods_per_year) * (excess_return.mean() / excess_return.std())

    def max_drawdown(capital_series):
        cumulative_max = capital_series.cummax()
        drawdown = (capital_series - cumulative_max) / cumulative_max
        return drawdown.min()
       
    def volatility(daily_return, periods_per_year=365):
        # daily_return = daily_return[daily_return != 0]
        return daily_return.std() * np.sqrt(periods_per_year)

    def adjust_for_fees(trades_df, fee_rate):
        trades_df['Fee Adjustment'] = trades_df['Remaining Capital'].diff().abs() * fee_rate
        trades_df['Net Remaining Capital'] = trades_df['Remaining Capital'] - trades_df['Fee Adjustment'].cumsum()
        return trades_df

    trades_df = adjust_for_fees(trades_df, fee_rate)

    daily_data = daily_data.loc[start_date:end_date]

    cta_annual_return = annualized_return(daily_data['CTA Daily Return'])
    benchmark_annual_return = annualized_return(daily_data['Benchmark Daily Return'])
    cta_sharpe = sharpe_ratio(daily_data['CTA Daily Return'])
    benchmark_sharpe = sharpe_ratio(daily_data['Benchmark Daily Return'])
    cta_max_drawdown = max_drawdown(daily_data['Remaining Capital'])
    benchmark_max_drawdown = max_drawdown(daily_data['Benchmark Capital'])
    cta_volatility = volatility(daily_data['CTA Daily Return'])
    benchmark_volatility = volatility(daily_data['Benchmark Daily Return'])

    performance_metrics = {
        'CTA': {
            'Annualized Return': cta_annual_return,
            'Sharpe Ratio': cta_sharpe,
            'Max Drawdown': cta_max_drawdown,
            'Volatility': cta_volatility,  
        },
        'Benchmark': {
            'Annualized Return': benchmark_annual_return,
            'Sharpe Ratio': benchmark_sharpe,
            'Max Drawdown': benchmark_max_drawdown,
            'Volatility': benchmark_volatility,  
        }
    }

    return pd.DataFrame(performance_metrics).round(4), daily_data

def plot_cumulative_returns(daily_data, holding_period):
    """
    Plot cumulative returns with final values added to the labels.

    Parameters:
    - daily_data: DataFrame containing daily returns and capital data.
    """
    # Calculate cumulative returns
    daily_data['CTA Cumulative Return'] = (1 + daily_data['CTA Daily Return']).cumprod() - 1
    daily_data['Benchmark Cumulative Return'] = (1 + daily_data['Benchmark Daily Return']).cumprod() - 1

    final_cta_return = daily_data['CTA Cumulative Return'].iloc[-1]
    final_benchmark_return = daily_data['Benchmark Cumulative Return'].iloc[-1]

    final_cta_return_label = f"CTA Strategy (Final: {final_cta_return:.2%})"
    final_benchmark_return_label = f"Benchmark (Final: {final_benchmark_return:.2%})"

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data.index, daily_data['CTA Cumulative Return'], label=final_cta_return_label, color='blue',linewidth=2)
    # plt.plot(daily_data.index, daily_data['Benchmark Cumulative Return'], label=final_benchmark_return_label, color='orange',linewidth=2)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add baseline
    plt.title(f'Cumulative Returns ({holding_period}-Day Holding Period)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


def backtest_zscore_strategy(results, initial_capital, rolling_window):
    """
    Backtest a z-score-based strategy where trades involve LL100, EE, and INFT.
    Returns:
        pd.DataFrame: DataFrame containing trade details.
    """
    trades = []
    capital = initial_capital
    position_active = False

    # Calculate rolling mean and standard deviation for residuals
    results['residuals_mean'] = results['residuals'].rolling(rolling_window).mean()
    results['residuals_std'] = results['residuals'].rolling(rolling_window).std()

    # Calculate z-score
    results['z_score'] = (results['residuals'] - results['residuals_mean']) / results['residuals_std']
    # Benchmark: Buy-and-hold strategy
    benchmark_shares = initial_capital / results['LL100'].iloc[0]  
    # Loop through each row in the DataFrame
    for i in range(len(results)):
        current_row = results.iloc[i]

        # If a position is active, check for closing condition
        if position_active:
            z_score = current_row['z_score']
            benchmark_capital = benchmark_shares * results['LL100'].iloc[i]  
            # Closing condition: -0.5 < z_score < 0.5
            if -0.5 < z_score < 0.5:
                # Calculate PnL for closing the position
                exit_price_LL100 = current_row['LL100']
                exit_price_EE = current_row['EE']
                exit_price_INFT = current_row['INFT']

                pnl_LL100 = (entry_price_LL100 - exit_price_LL100) * position_size_LL100 if entry_signal == -1 else \
                            (exit_price_LL100 - entry_price_LL100) * position_size_LL100
                pnl_EE = (exit_price_EE - entry_price_EE) * position_size_EE if entry_signal == -1 else \
                         (entry_price_EE - exit_price_EE) * position_size_EE
                pnl_INFT = (exit_price_INFT - entry_price_INFT) * position_size_INFT if entry_signal == -1 else \
                           (entry_price_INFT - exit_price_INFT) * position_size_INFT

                total_pnl = pnl_LL100 + pnl_EE + pnl_INFT
                capital += total_pnl

                # Record the trade
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_row.name,
                    'Entry Price LL100': entry_price_LL100,
                    'Exit Price LL100': exit_price_LL100,
                    'Entry Price EE': entry_price_EE,
                    'Exit Price EE': exit_price_EE,
                    'Entry Price INFT': entry_price_INFT,
                    'Exit Price INFT': exit_price_INFT,
                    'Position Size LL100': position_size_LL100,
                    'Position Size EE': position_size_EE,
                    'Position Size INFT': position_size_INFT,
                    'PnL': total_pnl,
                    'Remaining Capital': capital,
                    'Benchmark Capital': benchmark_capital 
                })

                # Close the position
                position_active = False

        # If no position is active, check for entry conditions
        if not position_active:
            z_score = current_row['z_score']
            benchmark_capital = benchmark_shares * results['LL100'].iloc[i]  
            if z_score > 1 or z_score < -1:
                # Entry condition met
                entry_price_LL100 = current_row['LL100']
                entry_price_EE = current_row['EE']
                entry_price_INFT = current_row['INFT']
                entry_signal = -1 if z_score > 1 else 1  # -1 for short, 1 for long

                # Allocate capital proportionally
                position_size_LL100 = capital / entry_price_LL100
                position_size_EE = 0.2 * capital / entry_price_EE
                position_size_INFT = 0.2 * capital / entry_price_INFT

                entry_date = current_row.name
                position_active = True

    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades)

    return trades_df, results


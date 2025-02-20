# -*- coding: utf-8 -*-

# In[1 - Imports]

import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# In[2 - SP Returns & Fetch Macro Factors]

# Initialize FRED API
FRED_API_KEY = 'd79a661d2578ede9498e265d7bb4a336' # user will have to input appropriate API key
fred = Fred(api_key=FRED_API_KEY)

# Function to fetch data from FRED
def fetch_fred_data(series_id, start_date='2000-01-01', end_date=None):
    """
    Fetch data from FRED using a series ID.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    data = fred.get_series(series_id, start=start_date, end=end_date)
    data = pd.DataFrame(data, columns=[series_id])
    data.index = pd.to_datetime(data.index)
    return data

# Fetch macroeconomic indicators
def fetch_macro_data(start_date='2000-01-01'):
    """
    Fetch and difference relevant macroeconomic data for S&P 500 predictions. 
    """
    macro_data = {}
    macro_data['DGS10'] = fetch_fred_data('DGS10', start_date)  # 10-Year Treasury Yield
    macro_data['DGS2'] = fetch_fred_data('DGS2', start_date)    # 2-Year Treasury Yield
    macro_data['M2SL'] = fetch_fred_data('M2SL', start_date) # M2 supply
    macro_data['CPILFESL'] = fetch_fred_data('CPILFESL', start_date)  # Core CPI
    macro_data['INDPRO'] = fetch_fred_data('INDPRO', start_date)  # Industrial Production Index
    macro_data['PAYEMS'] = fetch_fred_data('PAYEMS', start_date)  # Nonfarm Payrolls
    macro_data['VIXCLS'] = fetch_fred_data('VIXCLS', start_date)  # VIX Index
    macro_data['UMCSENT'] = fetch_fred_data('UMCSENT', start_date)  # Consumer Sentiment Index
    macro_data['DCOILWTICO'] = fetch_fred_data('DCOILWTICO', start_date)  # Crude Oil Prices
    macro_data['BUSINV'] = fetch_fred_data('BUSINV', start_date) # Business Inventories

    # Convert to DataFrame
    macro_df = pd.concat(macro_data.values(), axis=1)
    macro_df.columns = macro_data.keys()

    # Difference and drop null values (from mismatched dates)
    macro_df.dropna(inplace=True)
    macro_df = macro_df.pct_change(axis=0)
    macro_df.dropna(inplace=True)
    
    macro_df.index = macro_df.index.to_period('M')

    return macro_df

# Fetch S&P 500 Returns from flat file
def fetch_sp500_returns(start_date='2000-01-01', user_dir = r'C:\Users\Dimitris\Desktop\Financial Competency Assessment Deliverable'):
    """
    Fetch S&P 500 index values and compute monthly returns as of each month beginning to match FRED output.
    
    Parameters
    ----------
    user_dir (str): Flat file directory
    """
    import os
    # Change accordingly
    os.chdir(user_dir)  
    sp500_data = pd.read_excel(user_dir+'\DatasetExample.xlsx',sheet_name='Tab 2 (S&P 500 - daily)',skiprows=1)
    # Formatting
    sp500_data = sp500_data.iloc[1:,:2] # Exclude End of Dec value and null columns
    sp500_data.set_index('Dates',inplace=True)
            
    # Resample to monthly frequency and compute monthly returns
    sp500_data_monthly = sp500_data.resample('M').last()   # Get the end of month values
    sp500_data_monthly.index = sp500_data_monthly.index.to_period('M') # Reset date index to beginning of month for preper merging with macro data
    
    # Calc monthly returns
    sp500_data_monthly['Returns'] = sp500_data_monthly['S&P 500 Prices'].pct_change()
    sp500_data_monthly.dropna(inplace=True)
    
    return sp500_data_monthly[['Returns']]

# Merge S&P 500 with Macro Data
def create_dataset(start_date='2000-01-01'):
    """
    Create a dataset combining S&P 500 returns and macroeconomic indicators.
    """
    macro_df = fetch_macro_data(start_date)
    sp500_returns = fetch_sp500_returns(start_date)

    # Align datasets by index (date)
    dataset = sp500_returns.merge(macro_df, left_index=True, right_index=True, how='inner')
    return dataset

# In[3 - Train Random Forest Model]

# Prepare target variables
def create_lagged_targets(data, target_col='Returns', horizons=[1, 2, 3]):
    """
    Create lagged target variables for forecasting over specified horizons.
    
    Parameters
    ----------
    data (pd.DataFrame): The dataset containing the feature(s) and the target column
    target_col (str): Name of the column containing the dependent variable (e.g. S&P 500 returns)
    horizons (list of int): List of horizons (in months) for which lagged targets are created
    
    Returns
    -------
    (pd.DataFrame): Dataset with added lagged target variables
    """
    for horizon in horizons:
        data[f'Target_{horizon}M'] = data[target_col].shift(-horizon)
    # Drop nulls from lagging
    data.dropna(inplace=True)
    return data

# Train and evaluate Random Forest model
def train_random_forest(X, y, horizon):
    """
    Train a Random Forest model for a specific forecasting horizon

    Parameters
    ----------
    X (pd.DataFrame): Independent variables dataset
    y (pd.Series): Target dataset
    horizon (str): Forecasting horizon (e.g., "1M", "2M", or "3M")
        
    Returns
    -------
    RandomForestRegressor
    Predicted Values
    Realized Values
    RMSE
    R-Squared
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
    rf_model.fit(X_train, y_train)
    
    # Make predictions 
    y_pred = rf_model.predict(X_test)
    
    # Evaluate model with R^2 and RMSE 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = 1-np.sum(np.square(y_pred-y_test))/(np.sum(np.square(y_test-y_test.mean())))
   
    
    print(f"Horizon: {horizon}-Month")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print("-" * 30)
    
    return rf_model, y_pred, y_test, rmse, r2

# Main pipeline
def main_pipeline(dataset):
    """
    Main pipeline to train Random Forest models for 1-month, 2-month, and 3-month horizons.

    Parameters
    ----------
    dataset: pd.DataFrame
        The dataset containing feature variables and the target (S&P 500 monthly returns).

    Returns
    -------
    dict
        A dictionary of trained models for each forecasting horizon.
    """
    # Create lagged target variables
    dataset = create_lagged_targets(dataset, target_col='Returns', horizons=[1, 2, 3])
    
    # Define features and targets
    feature_columns = dataset.columns.difference(['Returns', 'Target_1M', 'Target_2M', 'Target_3M'])
    features = dataset[feature_columns]
    targets = {
        '1M': dataset['Target_1M'],
        '2M': dataset['Target_2M'],
        '3M': dataset['Target_3M']
    }
    
    # Ensure feature and target alignment
    models = {}
    results = {}
    for horizon, target in targets.items():
        print(f"Training model for {horizon} horizon...")
        model, y_pred, y_test, rmse, r2 = train_random_forest(features, target, horizon)
        
        # Store model and output parameters
        models[horizon] = model
        results[horizon] = {
            'predictions': y_pred,
            'actuals': y_test,
            'RMSE': rmse,
            'R2': r2
        }
    return models, results

# In[4 - Model Call]

# Random Forest model call
dataset = create_dataset()
models, results = main_pipeline(dataset)







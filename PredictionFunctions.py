import pandas as pd
import numpy as np
from sqlalchemy import create_engine as eng
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, mean_squared_error,r2_score


def LR_Predict_Prices_No_Predictors(df,version):
    # Split data using the static dates below
    train_start_date = datetime(2023, 1, 1)
    train_end_date = datetime(2023, 10, 1)
    
    train_data = df[(df['Activity_DTTM'] >= train_start_date) & (df['Activity_DTTM'] < train_end_date)]
    test_data = df[df['Activity_DTTM'] >= train_end_date]

    # Define features and target. X was created this way purposefully to create a far too basic model.
    X_train = train_data[['Activity_DTTM']]
    y_train = train_data['Activity_Close']

    X_test = test_data[['Activity_DTTM']]
    y_test = test_data['Activity_Close']

    # Convert dates to numerical representation (days since the start date)
    X_train.loc[:, 'Numeric_Date'] = (X_train['Activity_DTTM'] - X_train['Activity_DTTM'].min()) / np.timedelta64(1, 'D')
    X_test.loc[:, 'Numeric_Date'] = (X_test['Activity_DTTM'] - X_train['Activity_DTTM'].min()) / np.timedelta64(1, 'D')

    model = LinearRegression() 
    model.fit(X_train[['Numeric_Date']], y_train)
    # Make predictions on the test set
    test_predictions = model.predict(X_test[['Numeric_Date']])
    
    # Create a DataFrame with results
    results_df = pd.DataFrame({'Activity_DTTM': X_test['Activity_DTTM'],'Ticker': test_data['Ticker']
                               ,'Predicted_Stock_Price': test_predictions,'Version': version
                                })
    # Evaluate the model
    mse = mean_squared_error(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions)
    
    theta_values = model.coef_
    r_squared = r2_score(y_test, test_predictions)

    return results_df, mse, precision, theta_values, r_squared


def train_random_forest_regression(X):

    train_start_date = datetime(2023, 1, 1)
    train_end_date = datetime(2023, 10, 1)
    X = pd.DataFrame(X)
    X['Day'] = X['Activity_DTTM'].dt.day
    X['Month'] = X['Activity_DTTM'].dt.month
    X['Year'] = X['Activity_DTTM'].dt.year
    train_data = X[(X['Activity_DTTM'] >= train_start_date) & (X['Activity_DTTM'] < train_end_date)]
    test_data = X[X['Activity_DTTM'] >= train_end_date]
    
    # training data sets
    X_train = train_data
    X_train = X_train.drop(columns="Activity_DTTM")
    X_train = X_train.drop(columns="Ticker")

    y_train = train_data["Activity_Close"]
    #y_train = y_train.drop(columns="Activity_DTTM")

# testing data sets
    X_test = test_data
    X_test = X_test.drop(columns="Activity_DTTM")
    X_test = X_test.drop(columns="Ticker")
    x_test_dates = pd.DataFrame(test_data["Activity_DTTM"])

    y_test = test_data["Activity_Close"]
    

# Initialize and train the model
    trees = 200
    model = RandomForestRegressor(n_estimators=trees,random_state=1)
    model.fit(X_train, y_train)
    
    # Predict prices
    predicted_prices = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predicted_prices)

    predicted_prices = pd.DataFrame({
         'Activity_DTTM': x_test_dates['Activity_DTTM'],
         'Predicted_Stock_Price': predicted_prices,
     })
    
    return predicted_prices



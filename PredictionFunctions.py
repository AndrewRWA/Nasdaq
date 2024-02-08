import pandas as pd
import numpy as np
from sqlalchemy import create_engine as eng
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, mean_squared_error
#from sklearn.preprocessing import LabelEncoder


def LR_Predict_Prices_No_Predictors(data,version):

    df = pd.DataFrame(data)

    # Split data into train (3 months) and test (1 month)
    train_start_date = datetime(2023, 1, 1)
    train_end_date = datetime(2023, 10, 1)
    
    train_data = df[(df['Activity_DTTM'] >= train_start_date) & (df['Activity_DTTM'] < train_end_date)]
    test_data = df[df['Activity_DTTM'] >= train_end_date]

    # Define features and target
    X_train = train_data[['Activity_DTTM']]
    y_train = train_data['Activity_Close']

    X_test = test_data[['Activity_DTTM']]
    y_test = test_data['Activity_Close']

    # Convert dates to numerical representation (days since the start date)
    X_train.loc[:, 'Numeric_Date'] = (X_train['Activity_DTTM'] - X_train['Activity_DTTM'].min()) / np.timedelta64(1, 'D')
    X_test.loc[:, 'Numeric_Date'] = (X_test['Activity_DTTM'] - X_train['Activity_DTTM'].min()) / np.timedelta64(1, 'D')


    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train[['Numeric_Date']], y_train)
    # Make predictions on the test set
    test_predictions = model.predict(X_test[['Numeric_Date']])

    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'Activity_DTTM': X_test['Activity_DTTM'],
        'Ticker': test_data['Ticker'],
        #'Actual_Price': y_test,
        'Predicted_Stock_Price': test_predictions,
        'Version': version
    })

    # Evaluate the model
    mse = mean_squared_error(y_test, test_predictions)
    print(f'Mean Squared Error on Test Data: {mse}')

    return results_df 
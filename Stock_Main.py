from ModelSQLData import getDatatoModel, uploadtoSQL
from StatFunctions import calculate_coefficients,calculate_beta
from StockETL import getNasdaqTickers, downloadSQLdata, uploadNasdaqTickerDatatoSQL
from PredictionFunctions import LR_Predict_Prices_No_Predictors
import pandas as pd
import yfinance as yf
import numpy as np
from sqlalchemy import create_engine as eng
from datetime import datetime
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
import time
import msvcrt
from sklearn.metrics import precision_score, mean_squared_error
#from sklearn.preprocessing import LabelEncoder


#------------------------------------------------------------
#------------------------------------------------------------
# USER DEFINED FUNCTIONS (UDF)
def gettickerListforLoop(dataframe):
    #NasdaqTickers = ['abc123']
    NasdaqTickers = set(['abc123'])
    symbol_column_index = dataframe.columns.get_loc('Ticker')
    for symbol in dataframe.iloc[:,symbol_column_index]:
        #symbol = symbol_object['Symbol']
        NasdaqTickers.add(symbol)
        #NasdaqTickers.append(symbol)
    NasdaqTickers.remove("abc123")
    NasdaqTickerlist = list(NasdaqTickers).sort()
    
    return NasdaqTickerlist 
#------------------------------------------------------------

def timeout_input(prompt, timeout):
    start_time = time.time()
    user_input = None

    print(prompt)
    while time.time() - start_time < timeout:
        if msvcrt.kbhit():
            user_input = msvcrt.getch().decode('utf-8')
            break

    return user_input
#------------------------------------------------------------

def get_yfiTickerData(startDttm,Symbols):
    TickerData_objects = []
    
    for ticker in Symbols: 
        try:
            TickerData_object = yf.Ticker(ticker)
            historical_data = TickerData_object.history(start=startDttm)
            #historical_earnings = TickerData_object.earnings
            
            historical_data.index = historical_data.index.strftime('%Y-%m-%d %H:%M:%S')
            dttm = pd.to_datetime(historical_data.index, errors='coerce')

            historical_data.insert(0,'Activity_DTTM',dttm)
            historical_data.insert(1,'Ticker',ticker)
            historical_data = historical_data.dropna(subset=['Activity_DTTM'])

            if not historical_data.empty:
                TickerData_objects.append(historical_data)
                print(f"{ticker} completed")
            else:
                print("empty")

        except Exception as e:
            print(f"get_yfiTickerData() - Exception for ticker {ticker}: {str(e)}")

    df = pd.concat(TickerData_objects)

    return df

#------------------------------------------------------------
def get_yfiTickerIndustry(Symbols):    
    for ticker in Symbols: 
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            industry = info.get('industry', 'N/A')
            return industry
        
        except Exception as e:
            print(f"get_yfiTickerData() - Exception for ticker {ticker}: {str(e)}")
            return None

#------------------------------------------------------------
def predict(train,test,predictors,model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test[['Activity_DTTM', 'Target']],preds],axis=1)
    return combined
#------------------------------------------------------------

def predictProb(train,test,predictors,model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test[['Activity_DTTM', 'Target']],preds],axis=1)

    return combined
#------------------------------------------------------------

def backtest(data, model,predictors,version,start,step):
    all_predictions = []

    if version == min(LRMversion):
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions = predict(train,test,predictors,model)
            all_predictions.append(predictions)
    else:   
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions = predictProb(train,test,predictors,model)
            all_predictions.append(predictions)

    
    results = pd.concat(all_predictions)
    print(results)

    results_df = pd.DataFrame({
        'Activity_DTTM': results['Activity_DTTM'],
        'Ticker': ticker,
        'TargetBuySell': results['Target'],
        'Predictions': results['Predictions'],
        'Version': version
    })

    return results_df
#------------------------------------------------------------
# END USER DEFINED FUNCTIONS (UDF)
#------------------------------------------------------------


'''
-----------------------------------------------------------------------------------------------------
--------------------------------------BEGIN PROGRAM CODE---------------------------------------------
-----------------------------------------------------------------------------------------------------
'''


#--------------------------------------------------------------------
# Global Variables
startDttm       = datetime(2012, 1, 1)
LRMversion      = ["1.1.1","1.1.2"]
fetchNasdaqTickers = False
GetNewFinanceData = False
getIndustry = False
startLRMLoop = False
runCoeff = False
#--------------------------------------------------------------------



#--------------------------------------------------------------------
# Download Nasdaq Stock Ticker portfolio and push to SQL Server
if fetchNasdaqTickers:
    getNasdaqTickers(database='Nasdaq',schema='Dim',table_name='NasdaqStockTickers')

    # I RECOGNIZE THIS STEP IS UNNECESSARY/SKIPPABLE. The point is to show webscraping and database extraction capabilities
    NasdaqPortfolio_df = downloadSQLdata(database='Nasdaq',schema='Dim',table_name='NasdaqStockTickers') 
else:
    print("Dont fetch tickers list")
    NasdaqPortfolio_df = downloadSQLdata(database='Nasdaq',schema='Dim',table_name='NasdaqStockTickers') 
#--------------------------------------------------------------------



#--------------------------------------------------------------------
# FETCHING NEW yFinance DATA
# Set for ad-hoc purposes becaues data has been stored and purposefully remained static
NasdaqSymbols = sorted(NasdaqPortfolio_df['Symbol'].unique().tolist()) # Comprehensive List

getStart = '1990-01-01'
if GetNewFinanceData:
    yfiTickerData_Df = get_yfiTickerData(startDttm=getStart,Symbols=NasdaqSymbols)
    uploadNasdaqTickerDatatoSQL(importdf=yfiTickerData_Df,Database='Nasdaq',Schema='dbo',Table='NasdaqHistory')
else:
    print("Dont fetch ticker history from yFinance")
#--------------------------------------------------------------------
    


#--------------------------------------------------------------------
# Begin INTITIAL data modeling to predict if price will go up or down using RandomForestClassifier()
print('Retrieve data to model')
stockDataLRM    = getDatatoModel(database='Nasdaq',schema='dbo',table_name='NasdaqHistory',startDttm=startDttm )
# Setting aside for other models
modelData       = stockDataLRM
#--------------------------------------------------------------------



#--------------------------------------------------------------------
#Create interative list of ALL unique Nasdaq tickers
print('Set ticker list')
unique_Tickers = sorted(stockDataLRM['Ticker'].unique().tolist()) # Paired down subset intended for analysis
#--------------------------------------------------------------------



#--------------------------------------------------------------------
# Capture and upload industry data

if getIndustry:
    industry_data = []

    # Loop through each stock symbol to capture company industry
    for symbol in NasdaqSymbols:
        industry_info = get_yfiTickerIndustry(symbol)
        industry_data.append({'Symbol': symbol, 'Industry': industry_info})
        print(symbol)
    industry_df = pd.DataFrame(industry_data)
    print(industry_df)
    uploadtoSQL(importdf=industry_df,Database='Nasdaq',Schema='Dim',Table='NasdaqIndustries')
else:
    print("Skip industry")
#--------------------------------------------------------------------
    


#--------------------------------------------------------------------
# Begin Linear Regression Loop
print('Begin LR Model')
if startLRMLoop:
    for ticker in unique_Tickers: 
        df = stockDataLRM.loc[stockDataLRM['Ticker'] == ticker]

        try:
            predictedPrice = LR_Predict_Prices_No_Predictors(data=df,version=max(LRMversion))
            predictedPrice = predictedPrice.dropna(subset=['Activity_DTTM'])

            if not predictedPrice.empty:
                uploadtoSQL(importdf=predictedPrice,Database='Nasdaq',Schema='dbo',Table='NasdaqPredictedPrices')
                print(f"LRM Ticker: {ticker}, uploaded")
            else:
                print("Empty")
            
        except Exception as e:
            print(f"predict_tomorrows_stock_price() error: {e}")
else:
    print(f"Skip LRM")

# End Linear Regression Loop
#--------------------------------------------------------------------



#--------------------------------------------------------------------
# Begin Calculating Coefficients
print('Begin Corr Coeff')
if runCoeff:
    for ticker in unique_Tickers:
        filtered_data = stockDataLRM[stockDataLRM['Ticker'].isin(['QQQ', ticker])]
        del filtered_data['CapitalGains']
        del filtered_data['StockSplits']
        del filtered_data['Dividends']
        filtered_data.dropna()
        results = overall_coefficient = calculate_coefficients(filtered_data, ticker=ticker)
        if not results.empty and not ticker == "QQQ":
            uploadtoSQL(importdf=results,Database='Nasdaq',Schema='dbo',Table='NasdaqCoefficients')
            print(f"Coeff Ticker: {ticker}, uploaded")
        else:
            print("Empty")
else:
    print('Skip Coeff')
# End Calculating Coefficients
#--------------------------------------------------------------------



# Delete undesired columns from df
del modelData['Dividends']
del modelData['StockSplits']
del modelData['CapitalGains']

distinctTickerObject = []

for ticker in unique_Tickers:
        ticker_data = modelData[modelData["Ticker"] == ticker].copy()
        ticker_data['TomorrowPrice'] = ticker_data['Activity_Close'].shift(-1)

        # Set target to determine if we experienced a gain the next day
        ticker_data['Target'] = (ticker_data['TomorrowPrice'] > ticker_data['Activity_Close']).astype(int)

        distinctTickerObject.append(ticker_data)

        print(f"Created Tomorrow for: {ticker}")

print("Tomorrow Loop Completed")
modelData = pd.concat([modelData] + distinctTickerObject, ignore_index=True)
print("Tmr and ModelData concatenated")

modelData = modelData.dropna()
GainLossmodel = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)
modelTicker = "QQQ"

train = modelData[modelData["Ticker"] == modelTicker].iloc[:-200].dropna() # all but last 200 days
test = modelData[modelData["Ticker"] == modelTicker].iloc[-200:].dropna() #last 200 days


backtestPredict = modelData[modelData["Ticker"] == modelTicker].copy()
predictors = ['Activity_Close','Volume','Activity_Open','Activity_High','Activity_Low']
GainLossmodel.fit(train[predictors],train['Target'])
predictions = backtest(backtestPredict,GainLossmodel,predictors,min(LRMversion),2500,250)
uploadtoSQL(importdf=predictions,Database='Nasdaq',Schema='dbo',Table='NasdaqBuySell')
print(f'Total Predictions: \n', predictions['Predictions'].value_counts())
print(f'Prediction Score: \n', precision_score(predictions['TargetBuySell'],predictions['Predictions']))
print(f'Prediction Percent: \n',predictions["TargetBuySell"].value_counts() / predictions.shape[0])

# End INTITIAL data modeling to predict if price will go up or down using RandomForestClassifier()
#--------------------------------------------------------------------

#preds = model.predict(test[predictors])
#preds = pd.Series(preds, index=test.index)
#testTarget_Score = precision_score(test['Target'],preds)



#--------------------------------------------------------------------
# SECOND data modeling to predict if price will go up or down using RandomForestClassifier()
# with new predictors and using rolling time periods
print('Start second model')
periods = [2, 5, 60, 250] #,1000]
filtered_modelData = modelData[modelData["Ticker"] == modelTicker]
new_predictors = []
for period in periods:
    rolling_close_averages = filtered_modelData["Activity_Close"].rolling(period).mean()
    rolling_volume_averages = filtered_modelData["Volume"].rolling(period).mean()

    volume_column = f"Volume_Ratio_{period}_Days"
    filtered_modelData[volume_column] = filtered_modelData["Volume"] / rolling_volume_averages

    ratio_column = f"Close_Ratio_{period}_Days"
    filtered_modelData[ratio_column] = filtered_modelData["Activity_Close"] / rolling_close_averages

    trend_column = f"Trend_{period}_Days"
    filtered_modelData[trend_column] = filtered_modelData["Target"].shift(1).rolling(period).sum()

    new_predictors += [ratio_column, trend_column, volume_column]


secondPrediction = filtered_modelData.dropna()
Secondmodel = RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=1)


Secondmodel.fit(train[predictors],train['Target'])
SecondPredictions = backtest(secondPrediction,Secondmodel,new_predictors,max(LRMversion),2500,250)
print(f'Total Predictions: \n', SecondPredictions['Predictions'].value_counts())
print(f'Prediction Score: \n', precision_score(SecondPredictions['TargetBuySell'],SecondPredictions['Predictions']))
print(f'Prediction Percent: \n',SecondPredictions["TargetBuySell"].value_counts() / SecondPredictions.shape[0])
print(f'Records in prediction data: ', secondPrediction.shape[0])
uploadtoSQL(importdf=SecondPredictions,Database='Nasdaq',Schema='dbo',Table='NasdaqBuySell')

'''
model = RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=1)
backtestPredict = secondPrediction[secondPrediction["Ticker"] == modelTicker].copy()

model.fit(train[predictors],train['Target'])
SecondPredictions = backtest(secondPrediction,GainLossmodel,new_predictors,2,250,25)


# Begin SECOND data modeling to predict if price will go up or down using RandomForestClassifier()

columns_to_delete = ['Activity_Open', 'Activity_Close','Activity_High','Activity_Low','Volume']
secondPrediction = secondPrediction.drop(columns=columns_to_delete)
print(f'Second Prediction results\n')
print(SecondPredictions["Predictions"].value_counts())
print(precision_score(SecondPredictions["Target"],SecondPredictions['Predictions']))
'''
# End SECOND data modeling to predict if price will go up or down using RandomForestClassifier()
#--------------------------------------------------------------------


#df = stockData.drop(columns="Ticker").copy()
#--------------------------------------------------------------------
# Predict stock prices using linear regression
#TickerPrice_objects = []





#df = df.drop(columns="Ticker")


#file_path = 'C:/Users/Andre/OneDrive/Documents/myexcel.xlsx'

#sheet_name = 'Sheet1'
#df.to_excel(file_path,sheet_name=sheet_name, index=False)
#quit()






'''
# Plot pre-modeled data
#modelData.plot.line(y='Activity_Close', x='Activity_DTTM', legend='Ticker')
selected_tickers = ['AAPL', 'GOOGL', 'MSFT','QQQ']
# Group by 'Ticker' and calculate the mean of 'Activity_Close' for each group
grouped_data = modelData.groupby(['Ticker', 'Activity_DTTM'])['Activity_Close'].mean().reset_index()

# Plotting each ticker
#for ticker in grouped_data['Ticker'].unique()[:10]:
for ticker in selected_tickers:
    ticker_data = grouped_data[grouped_data['Ticker'] == ticker]
    plt.plot(ticker_data['Activity_DTTM'], ticker_data['Activity_Close'], label=ticker)

plt.xlabel('Activity_DTTM')
plt.ylabel('Activity_Close')
plt.title('Daily Activity Close for Each Ticker')
plt.legend()
plt.show()
'''
import pandas as pd
from sqlalchemy import create_engine as eng
from datetime import datetime


# Create connection to SQL Server
'''
database            = 'Dimensions'
schema              = 'Dim'
table_name          = 'NasdaqStockTickers'
'''


def getNasdaqTickers(database,schema,table_name):
    #conn = sqlite3.connect('ANDREW_PC\ANDREWSSQLSEVER.Dimensions')
    #cursor = conn.cursor()
    try:
        server              = 'ANDREW_PC\ANDREWSSQLSEVER'
        trusted_connection  = 'yes'
        DimensionsEngine    = eng(f'mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver=ODBC+Driver+17+for+SQL+Server&schema={schema}')
    
        # Download Nasdaq Stock Ticker portfolio and push to SQL Server
        nasdaq_symbols = pd.read_csv('ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt', sep='|')

        nasdaq_symbols.to_sql(schema=schema,name=table_name, con=DimensionsEngine, if_exists='replace', index=False)

    except Exception as e:
        print(f"Error Downloading Nasdaq Symbols: {e}")

    print('getNasdaqTickers() Completed')

def downloadSQLdata(database,schema,table_name):
    server              = 'ANDREW_PC\ANDREWSSQLSEVER'
    trusted_connection  = 'yes'
    DimensionsEngine    = eng(f'mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver=ODBC+Driver+17+for+SQL+Server&schema={schema}')
    
    query = f'''SELECT 
                    RecordID = ROW_NUMBER() over(order by Symbol)
                    ,* 
                FROM {schema}.{table_name} 
                WHERE Symbol is not null
                ORDER BY Symbol'''

    # Execute the query and fetch data into a DataFrame
    try:
        data = pd.read_sql(sql=query, con=DimensionsEngine)
    except Exception as e:
        print(f"Error: {e}")
        data = None

    return data



def uploadNasdaqTickerDatatoSQL(importdf,Database,Schema,Table):

    starttime           = datetime.now()
    table_exists        = 1
    server              = 'ANDREW_PC\ANDREWSSQLSEVER'
    trusted_connection  = 'yes'
    NasdaqEngine        = eng(f'mssql+pyodbc://{server}/{Database}?trusted_connection={trusted_connection}&driver=ODBC+Driver+17+for+SQL+Server&schema={Schema}')

    column_mapping = {
                       'Open': 'Activity_Open'
                      , 'High': 'Activity_High'
                      , 'Low': 'Activity_Low'
                      , 'Close': 'Activity_Close'
                      , 'Volume': 'Volume'
                      , 'Dividends': 'Dividends'
                      , 'Stock Splits': 'StockSplits'
                      , 'Ticker': 'Ticker'
                      , 'Date': 'Activity_DTTM'
                      , 'Capital Gains': 'CapitalGains'
                      }
    df_mapped = importdf.rename(columns=column_mapping)
    are_equal = False
    try: # ...pushing delta load of Nasdaq Stock Ticker portfolio to SQL Server
        query = f"select Ticker,MAX(Activity_DTTM) as MaxRecordedDateTime from {Database}.{Schema}.{Table} GROUP BY Ticker"
        existing_data = pd.read_sql(query, NasdaqEngine)
        max_values_df = df_mapped.loc[df_mapped.groupby('Ticker')['Activity_DTTM'].idxmax()]
        are_equal = existing_data.equals(max_values_df)
        # Identify new and modified records in the DataFrame
        new_records = df_mapped.merge(existing_data[['Ticker', 'MaxRecordedDateTime']], on=['Ticker', 'MaxRecordedDateTime'], how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    except Exception as e:
        print(f"Delta Load Error: {e}")
        table_exists = 0

    
    if are_equal:
        print("No Changes to Data")
    elif table_exists ==0:
        print("New data but table does not exist")
        try:  # ...pushing full Nasdaq Stock Ticker portfolio to SQL Server
            
            localdf = df_mapped
            localdf.to_sql(schema=Schema,name=Table, con=NasdaqEngine, if_exists='replace', index=False)
            enddatetime =datetime.now()
            timedifferencial = enddatetime - starttime
            print(f"uploadNasdaqTickerDatatoSQL() replace completed in {timedifferencial.total_seconds() / 60}")
        except Exception as e:
            print(f"Error Uploading Nasdaq Data: {e}")
    else:
        new_records.to_sql(schema=Schema,name=Table, con=NasdaqEngine, if_exists='append', index=False)
        timedifferencial = enddatetime - starttime
        enddatetime =datetime.now()
        print(f"uploadNasdaqTickerDatatoSQL() append completed in {timedifferencial.total_seconds() / 60}")
# -- End uploadNasdaqTickerDatatoSQL()



    
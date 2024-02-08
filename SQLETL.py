import pandas as pd
from sqlalchemy import create_engine as eng, MetaData
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

def does_table_exists(engine, table_name):
    meta = MetaData()
    meta.reflect(bind=engine)
    return table_name in meta.tables

def getDatatoModel(database,schema,table_name,startDttm):
    server              = 'ANDREW_PC\ANDREWSSQLSEVER'
    trusted_connection  = 'yes'
    Engine = eng(f'mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver=ODBC+Driver+17+for+SQL+Server&schema={schema}')

    def get_date_value(query):
        try:
            data = pd.read_sql(sql=query, con=Engine)
        except Exception as e:
            print(f"Error: {str(e)}")
        return data.iloc[0, 0] if not data.empty else None
    
    minDatequery = f"SELECT min(Activity_DTTM) as minActivity_DTTM FROM {database}.{schema}.{table_name} where Activity_DTTM >= '{startDttm}' "
    minDateValue = get_date_value(minDatequery)
    maxDatequery = f"SELECT max(Activity_DTTM) as maxActivity_DTTM FROM {database}.{schema}.{table_name} where Activity_DTTM >= '{startDttm}' "
    maxDateValue = get_date_value(maxDatequery)

    if minDateValue.year is None:
        datecounter = startDttm.year
    else:
        datecounter = 2000

    datecounter = startDttm.year
    query_results_object = []
    while datecounter <= maxDateValue.year:
        query = f"""
                    with cte as (
                            select ticker,count(1) as cnt
                            from [Nasdaq].[dbo].[NasdaqHistory]
                            where activity_dttm >= '1/1/2010'
                            group by ticker
                            having count(1) >= 2500 and max(activity_dttm) >= '2024-01-10'
                            )
                    SELECT a.* 
                    FROM {database}.{schema}.{table_name} as a 
                    join cte as b 
                    on a.ticker = b.ticker 
                    where year(Activity_DTTM) = '{datecounter}' """
        data = pd.read_sql(sql=query,con=Engine)
        query_results_object.append(data)
        print("Execution results for: ", datecounter, " with ", data.shape[0], " size")
        #datecounter += timedelta(days=1)
        datecounter += 1
        
    df = pd.concat(query_results_object)
    df_copy = df.reset_index()
    print(f"Shape: {df.shape[0]}")
    return df

def uploadtoSQL(importdf, Database,Schema,Table):

    starttime           = datetime.now()
    server              = 'ANDREW_PC\ANDREWSSQLSEVER'
    database            = Database #'Nasdaq'
    schema              = Schema #'dbo'
    table_name          = Table #'NasdaqModeledData'
    trusted_connection  = 'yes'
    NasdaqEngine        = eng(f'mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver=ODBC+Driver+17+for+SQL+Server&schema={schema}')

    #new_records = importdf.merge(existing_data[['Ticker', 'MaxRecordedDateTime']], on=['Ticker', 'Activity_DTTM'], how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    importdf.to_sql(schema=schema,name=table_name, con=NasdaqEngine, if_exists='append', index=False)
    #enddatetime =datetime.now()
    #timedifferential = enddatetime - starttime
    #print(f"uploadModeltoSQL() append completed in {timedifferential.total_seconds() / 60}")


# -- End uploadModeltoSQL()
    









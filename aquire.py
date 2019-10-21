# Goal: leave this section with a dataframe ready to prepare.

# The ad hoc part includes summarizing your data as you read it in 
# and begin to explore, look at the first few rows, data types, 
# summary stats, column names, shape of the data frame, etc.

# acquire.py: The reproducible part is the gathering data from SQL.

import pandas as pd
import numpy as np

import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def aquire_database():
    query = '''
    select *
    from properties_2017
    limit 10;
    '''
    df = pd.read_sql(query, get_db_url('zillow'))
    return df


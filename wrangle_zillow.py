import pandas as pd
import numpy as np

import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_fips_number_from_mysql():
    query = '''
    select fips
    from properties_2017
    group by fips;
    '''

    df = pd.read_sql(query, get_db_url('zillow'))

    return df

def clean_fips_data(df):
    df['fips_number'] = df.fips
    df = df.drop(columns=['fips'])
    df = df.dropna()
    df.fips_number = df.astype(int)
    df['County'] = ['Los Angeles County', 'Orange County', 'Ventura County']
    df = df.set_index('fips_number')
    return df
  
def wrangle_county_fips():
    df = get_fips_number_from_mysql()
    df = clean_fips_data(df)
    return df


def get_zillow_data_from_mysql():
    query = '''
    SELECT bedroomcnt as bedrooms,
       bathroomcnt as bathrooms,
       calculatedfinishedsquarefeet as square_feet,
       taxamount as property_tax,
       taxvaluedollarcnt as house_value,
       propertylandusedesc as property_description,
       propertylandusetypeid property_id,
       fips
    FROM predictions_2017
    JOIN properties_2017 USING(id)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE (transactiondate >= '2017-05-01' AND transactiondate <= '2017-06-30') 
    AND propertylandusetypeid = '261' 
    OR (propertylandusetypeid = '279' AND propertylandusedesc='Single Family Residential')
    ORDER BY fips;
    '''

    df = pd.read_sql(query, get_db_url('zillow'))
    return df

def clean_zillow_data(df):
    df = df.replace({'fips': 6037.0}, 'Los Angeles County')
    df = df.replace({'fips': 6059.0}, 'Orange County')
    df = df.replace({'fips': 6111.0}, 'Ventura County')
    df = df.dropna()
    return df
  
def wrangle_zillow_data():
    df = get_zillow_data_from_mysql()
    df = clean_zillow_data(df)
    return df
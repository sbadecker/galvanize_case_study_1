import numpy as np
import pandas as pd
import datetime as dt


'''
df = pd.read_csv('data/Train.csv')

year = df['YearMade']
year = year[year != 1000]

price_v_year = df[['SalePrice', 'YearMade']]
'''

def load_data():
    df = pd.read_csv('../data/train.zip', compression='zip')
    df['saledate'] = pd.to_datetime(df['saledate'], infer_datetime_format=True)
    df['saleyear'] = df['saledate'].dt.year
    df['ModelID'] = df['ModelID'].astype(str)
    df = df[df['YearMade'] != 1000]
    df['age'] = df['saleyear'] - df['YearMade']
    df = df[df['age'] >= 0]
    data = df[['ProductGroup','age', 'ProductSize','fiBaseModel','YearMade','ModelID','saleyear','UsageBand','state','SalePrice']]
    return data

if __name__ == '__main__':
    data = load_data()

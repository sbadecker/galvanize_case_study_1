import numpy as np
import pandas as pd
import datetime as dt

from zipfile import ZipFile

zf = ZipFile('data/Train.zip')

'''
df = pd.read_csv('data/Train.csv')

year = df['YearMade']
year = year[year != 1000]

price_v_year = df[['SalePrice', 'YearMade']]
'''

def load_data():
    #df = pd.read_csv('../data/train.csv',parse_dates=True)
    df['saledate'] = pd.to_datetime(df['saledate'], infer_datetime_format=True)
    df['saleyear'] = df['saledate'].dt.year
    data = df[['ProductGroup','ProductSize','fiBaseModel','YearMade','ModelID','saledate','saleyear','UsageBand','state','SalePrice']]
    return data

if __name__ == '__main__':
    data = load_data()

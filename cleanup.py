import numpy as np
import pandas as pd
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


'''
df = pd.read_csv('data/Train.csv')

year = df['YearMade']
year = year[year != 1000]

price_v_year = df[['SalePrice', 'YearMade']]
'''


def load_data(max_rows=1000):
    df = pd.read_csv('../data/train.zip', compression='zip')[:max_rows]
    df['saledate'] = pd.to_datetime(df['saledate'], infer_datetime_format=True)
    df['saleyear'] = df['saledate'].dt.year
    df['ModelID'] = df['ModelID'].astype(str)
    df = df[df['YearMade'] != 1000]
    df['age'] = df['saleyear'] - df['YearMade']
    df = df[df['age'] >= 0]
    data = df[['ProductGroup','age', 'ProductSize','fiBaseModel','YearMade','ModelID','saleyear','UsageBand','SalePrice']]
    pd.get_dummies(df, drop_first=True, columns=['ProductGroup','ProductSize','fiBaseModel','ModelID','UsageBand'])
    return data

if __name__ == '__main__':
    data = load_data()
    y = data.pop('SalePrice')[:1000]
    X = data[:1000]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rand_forest = RandomForestClassifier()
    rand_forest.fit(X_train, y_train)
    print rand_forest.score(X_test, y_test)

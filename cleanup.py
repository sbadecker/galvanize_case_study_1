import numpy as np
import pandas as pd
import datetime as dt
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV


'''
df = pd.read_csv('data/Train.csv')

year = df['YearMade']
year = year[year != 1000]

price_v_year = df[['SalePrice', 'YearMade']]
'''


def load_data(max_rows=10000):
    df = pd.read_csv('../data/train.zip', compression='zip')
    df['saledate'] = pd.to_datetime(df['saledate'], infer_datetime_format=True)
    df['saleyear'] = df['saledate'].dt.year
    df['ModelID'] = df['ModelID'].astype(str)
    df = df[df['YearMade'] != 1000]
    df['age'] = df['saleyear'] - df['YearMade']
    df = df[df['age'] >= 0]
    df['age2'] = df['age']**2
    # data = df[['ProductGroup','age', 'ProductSize','fiBaseModel','YearMade','ModelID','saleyear','UsageBand','SalePrice']]
    data = df[['ProductGroup','age', 'age2', 'ProductSize','SalePrice', 'YearMade','fiBaseModel']]
    data = pd.get_dummies(data, drop_first=True, columns=['ProductGroup','ProductSize','fiBaseModel'])
    return data

if __name__ == '__main__':
    data = load_data()
    y = data.pop('SalePrice')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rand_forest = RandomForestRegressor(n_estimators=40, oob_score=True, n_jobs=4)
    rand_forest.fit(X_train, y_train)
    print rand_forest.score(X_test, y_test)
    for model in [LassoCV, RidgeCV, LinearRegression]:
         mod = model()
         mod.fit(X_train, y_train)
         print str(model), mod.score(X_test, y_test)

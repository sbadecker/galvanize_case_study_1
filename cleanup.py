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


def load_data(filename, max_rows=10000):
    df = pd.read_csv(filename, compression='zip')
    df['saledate'] = pd.to_datetime(df['saledate'], infer_datetime_format=True)
    df['saleyear'] = df['saledate'].dt.year
    df['ModelID'] = df['ModelID'].astype(str)
    df = df[df['YearMade'] != 1000]
    df['age'] = df['saleyear'] - df['YearMade']
    df = df[df['age'] >= 0]
    # data = df[['ProductGroup','age', 'ProductSize','fiBaseModel','YearMade','ModelID','saleyear','UsageBand','SalePrice']]
    data = df[['ProductGroup','age', 'ProductSize','SalePrice', 'YearMade','fiBaseModel']]
    data = pd.get_dummies(data, drop_first=True, columns=['ProductGroup','ProductSize','fiBaseModel'])
    return data

def load_test_data(filename, max_rows=10000):
    df = pd.read_csv(filename, compression='zip')
    df['saledate'] = pd.to_datetime(df['saledate'], infer_datetime_format=True)
    df['saleyear'] = df['saledate'].dt.year
    df['ModelID'] = df['ModelID'].astype(str)
    df['age'] = df['saleyear'] - df['YearMade']
    # data = df[['ProductGroup','age', 'ProductSize','fiBaseModel','YearMade','ModelID','saleyear','UsageBand','SalePrice']]
    data = df[['ProductGroup','age', 'ProductSize', 'YearMade','fiBaseModel']]
    data = pd.get_dummies(data, drop_first=True, columns=['ProductGroup','ProductSize','fiBaseModel'])
    return data

if __name__ == '__main__':
    data = load_data('../data/train.zip')[:1000]
    test_data = load_test_data('../data/test.zip')[:1000]
    y = data.pop('SalePrice')
    X = data
    X_final = test_data
    rand_forest = RandomForestRegressor(n_estimators=250, oob_score=True, n_jobs=4)
    rand_forest.fit(X, y)
    y_pred = rand_forest.predict(X_final)
    final_df = pd.DataFrame([test_data['SalesID'],y_pred])
    final_df.to_csv('output.csv')
    # rand_forest.fit(X_train, y_train)

    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # rand_forest = RandomForestRegressor(n_estimators=400, oob_score=True, n_jobs=4)
    # rand_forest.fit(X_train, y_train)
    #
    # print rand_forest.score(X_test, y_test)
    # for model in [LassoCV, RidgeCV, LinearRegression]:
    #      mod = model()
    #      mod.fit(X_train, y_train)
    #      print str(model), mod.score(X_test, y_test)

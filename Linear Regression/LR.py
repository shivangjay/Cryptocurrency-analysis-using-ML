import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import datetime as dt
from sklearn.metrics import mean_squared_error, r2_score
files=["BTCUSD.csv","DASHUSD.csv","DOGEUSD.csv","ETHUSD.csv","LTCUSD.csv","XMRUSD.csv"];
for i in files:
    df = pd.read_csv(i)
    print(df.head())
    df = df[['Date','Close']] 
    df['Date'] = pd.to_datetime(df.Date,format='%d/%m/%Y')
    df['Date']=df['Date'].map(dt.datetime.toordinal)
    print(df.head())
    X = df.iloc[:, 0].values.reshape(-1, 1)  
    Y = df.iloc[:, 1].values.reshape(-1, 1) 
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train, y_train)
    y_pred = linear_regressor.predict(x_test)
    plt.scatter(x_test, y_test,  color='green')
    plt.plot(x_test, y_pred, color='red', linewidth=4)
    plt.title(i)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price(in USD)', fontsize=14)
    plt.grid(True)
    print('Mean squared error: %.2f' % mean_squared_error(y_test,y_pred))
    print('Coefficient of determination %.2f'% r2_score(y_test, y_pred))
    plt.show()

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    return (df - df.min()) / (data.max() - df.min())
def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)
def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',dropout=0.2, loss='mse', optimizer='eve'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('Close Price (in USD)', fontsize=14)
    ax.set_xlabel('Time(Indexed)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);
    plt.show()

files=["BTCUSD.csv","DASHUSD.csv","DOGEUSD.csv","ETHUSD.csv","LTCUSD.csv","USD.csv"];
for i in files:
    np.random.seed(42)
    window_len = 5
    test_size = 0.2
    zero_base = True
    lstm_neurons = 100
    epochs = 20
    batch_size = 32
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'
    hist = pd.read_csv(i)
    hist = hist[['Date','Close']]
    hist['Date'] = pd.to_datetime(hist.Date,format='%d/%m/%Y')
    hist['Date']=hist['Date'].map(dt.datetime.toordinal)
    target_col = 'Close'
    train, test = train_test_split(hist, test_size=0.2)
    line_plot(train[target_col], test[target_col], 'training', 'test', title=i.split('.')[1])
    train, test, X_train, X_test, y_train, y_test = prepare_data(hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    targets = test[target_col][window_len:]
    preds = model.predict(X_test).squeeze()
    mean_absolute_error(preds, y_test)
    preds = test[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, p
              
              for i in files:

    np.random.seed(42)

    window_len = 5

    test_size = 0.2

    zero_base = True

    lstm_neurons = 100

    epochs = 20

    batch_size = 32

    loss = 'mse'

    dropout = 0.2

    optimizer = 'eve'

    hist = pd.read_csv(i)

    hist = hist[['Date','Close']]

    hist['Date'] = pd.to_datetime(hist.Date,format='%d/%m/%Y')

    hist['Date']=hist['Date'].map(dt.datetime.toordinal)

    target_col = 'Close'

    train, test = train_test_split(hist, test_size=0.2)

    line_plot(train[target_col], test[target_col], 'training', 'test', title=i.split('.')[0])

    train, test, X_train, X_test, y_train, y_test = prepare_data(hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,optimizer=optimizer)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    targets = test[target_col][window_len:]

    preds = model.predict(X_test).squeeze()

    mean_absolute_error(preds, y_test)

    preds = test[target_col].values[:-window_len] * (preds + 1)

    preds = pd.Series(index=targets.index, data=preds)

    line_plot(targets, preds, 'actual', 'prediction', i.split('.')[0], lw=3)

reds, 'actual', 'prediction', i.split('.')[0], lw=3)

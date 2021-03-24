import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from math import sqrt
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf


def read_raw():
    dataset = pd.read_csv('../train_deal_c/train/data_repair/data_repair_mean2.csv')
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    dataset.set_index('Timestamp')
    dataset.index.name = 'date'
    dataset = dataset[24:]
    dataset = dataset[['Value']]
    dataset.to_csv('../train_deal_c/train/data_repair/data_repair_mean2_lstm.csv')



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def cs_to_sl():
    dataset = pd.read_csv('../train_deal_c/train/data_repair/data_repair_mean2_lstm.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 24, 1)
    reframed.to_csv('../train_deal_c/train/data_repair/data_repair_mean2_lstm2.csv')
    return reframed, scaler


def train_test(reframed):
    values = reframed.values
    n_train_hours = 365 * 24 * 2
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 24, 1))
    test_X = test_X.reshape((test_X.shape[0], 24, 1))
    return train_X, train_y, test_X, test_y


def fit_network(train_X, train_y, test_X, test_y, scaler):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=1, min_lr=0.001)
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=20, batch_size=64, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False, callbacks=[reduce_lr])
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
    #inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    #inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = scaler.inverse_transform(yhat)
    inv_y = test_y.reshape(-1, 1)
    inv_y = scaler.inverse_transform(inv_y)
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    return model


if __name__ == '__main__':
    read_raw()
    reframed, scaler = cs_to_sl()
    train_X, train_y, test_X, test_y = train_test(reframed)
    model = fit_network(train_X, train_y, test_X, test_y, scaler)
    model.save('../models/lstm0.tf')

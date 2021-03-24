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

def train_test(reframed):
    values = reframed.values
    n_train_hours = 30000
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
    history = model.fit(train_X, train_y, epochs=300, batch_size=64, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False, callbacks=[reduce_lr])
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig('../epochs_pic/train_cutting2_lstm_mean.png')
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
    dataset = pd.read_csv('../train_cuting/train_cutting2_lstm_mean.csv')
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    dataset = dataset.set_index('Timestamp')
    dataset.index.name = 'date'
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = dataset['Value']
    values = values.values
    values = values.reshape(-1, 1)
    values = scaler.fit_transform(values)
    reframed = pd.read_csv('../train_deal_c/train/data_repair/data_repair_mean2_lstm_group.csv', index_col=0)
    train_X, train_y, test_X, test_y = train_test(reframed)
    model = fit_network(train_X, train_y, test_X, test_y, scaler)
    model.save('../models/lstm1.tf')
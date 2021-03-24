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
from lstm2 import series_to_supervised

hours = 200
start = 364*24*3
model = tf.keras.models.load_model('../models/lstm0.tf')
dataset = pd.read_csv('../train_deal_c/train/data_repair/data_repair_mean2_lstm.csv', index_col=0)
scaler = MinMaxScaler(feature_range=(0, 1))
values = dataset.values
values = values.astype('float32')
y = values[start+hours:start+2*hours]
values = scaler.fit_transform(values)
values = series_to_supervised(values, 24, 1)
values = values.values
predictions = values[start+hours:start+2*hours, :-1]
predictions_X = predictions.reshape(predictions.shape[0], 24, 1)
yhat = model.predict(predictions_X)
predictions_X = predictions_X.reshape(predictions_X.shape[0], 24)
yhat = scaler.inverse_transform(yhat)

print(y)
print(yhat)

rmse = sqrt(mean_squared_error(y, yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(yhat, label='yhat', color='red')
pyplot.plot(y, label='y', color='green')
pyplot.savefig('../pre_picture/1.png')
pyplot.show()
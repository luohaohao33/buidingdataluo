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

hours = 200
start = 5500
model = tf.keras.models.load_model('../models/lstm1.tf')
dataset = pd.read_csv('../test_deal_c/test/data_repair/data_repair_mean2_lstm_group.csv', index_col=0)

data = pd.read_csv('../train_cuting/train_cutting2_lstm_mean.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
values = data['Value']
values = values.values
values = values.reshape(-1, 1)
scaler.fit_transform(values)

dataset = dataset.values
predictions = dataset[start+hours:start+2*hours, :-1]
predictions_X = predictions.reshape(predictions.shape[0], 24, 1)
y_1 = dataset[start+hours:start+2*hours, -1]
yhat_1 = model.predict(predictions_X)
#print(y)
#print(yhat)
y_1 = y_1.reshape(-1, 1)
yhat_1 = yhat_1.reshape(-1, 1)
y = scaler.inverse_transform(y_1)
yhat = scaler.inverse_transform(yhat_1)

rmse = sqrt(mean_squared_error(y, yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(yhat, label='yhat', color='red')
pyplot.plot(y, label='y', color='green')
pyplot.savefig('../pre_picture/lstm3_cutting2.png')
pyplot.show()
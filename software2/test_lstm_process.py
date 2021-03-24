import pandas as pd
import numpy as np
from util import *

if __name__ == '__main__':
    #数据处理1
    #TV_data('../test_cuting', 'test_cutting2.csv', '../test_cuting', 'test_cutting2_lstm.csv')
    #mean('../test_cuting', 'test_cutting2_lstm.csv', '../test_cuting', 'test_cutting2_lstm_mean.csv')
    #grouped('../test_cuting', 'test_cutting2_lstm_mean.csv', '../test_cuting', 'test_cutting2_lstm_mean.csv')
    data = pd.read_csv('../train_cuting/train_cutting2_lstm_mean.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = data['Value']
    values = values.values
    values = values.reshape(-1, 1)
    scaler.fit_transform(values)

    data_process_lstm('../test_cuting', 'test_cutting2_lstm_mean.csv', '../test_deal_c/test/data_repair',
                      'data_repair_mean2_lstm_group.csv', scaler)



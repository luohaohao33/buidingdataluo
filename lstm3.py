import pandas as pd

from sklearn.preprocessing import MinMaxScaler



def data_process():
    dataset = pd.read_csv('../train_cuting/train_cutting2_lstm_mean.csv')
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    dataset = dataset.set_index('Timestamp')
    dataset.index.name = 'date'
    len = dataset['groups'].max()
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = dataset['Value']
    values = values.values
    values = values.reshape(-1, 1)
    values = scaler.fit_transform(values)
    dataset['Value'] = values
    #print(dataset)
    for i in range(1, len+1):
        temp_data = dataset[dataset['groups'] == i]
        temp_data = temp_data['Value']
        temp_values = temp_data.values
        temp_values = temp_values.astype('float32')
        temp_values = temp_values.reshape(-1, 1)
        reframed = series_to_supervised(temp_values, 24, 1)
        if i == 1:
            set = reframed
        else:
            set = pd.concat([set, reframed])

    #set.to_csv('../train_deal_c/train/data_repair/data_repair_mean2_lstm_group.csv')
    print(set)





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


if __name__ == '__main__':
    data_process()
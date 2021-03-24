import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def scal():
    dataset = pd.read_csv('../train_cuting/train_cutting2_lstm_mean.csv')
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    dataset = dataset.set_index('Timestamp')
    dataset.index.name = 'date'
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = dataset['Value']
    values = values.values
    values = values.reshape(-1, 1)
    scaler.fit_transform(values)
    return scaler


def data_process_lstm(path1=None, name1=None, path2=None, name2=None, scaler=None):
    if path1 == None:
        dataset = pd.read_csv(name1)
    else:
        dataset = pd.read_csv(path1 + '/' + name1)
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    dataset = dataset.set_index('Timestamp')
    dataset.index.name = 'date'
    len = dataset['groups'].max()
    values = dataset['Value']
    values = values.values
    values = values.reshape(-1, 1)
    values = scaler.fit_transform(values)
    dataset['Value'] = values
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
    if path2 == None:
        set.to_csv(name2)
    else:
        set.to_csv(path2 + '/' + name2)





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


def mean(path1, name1, path2, name2):
    df = pd.read_csv(path1 + '/' + name1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    hour = pd.Timedelta('1h')
    dt = df['Timestamp']
    in_block = (dt.diff() == hour)
    in_block[0] = True

    temp_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    temp_mean_imputer.fit(df[['Value']])
    df['Value'] = temp_mean_imputer.transform(df[['Value']])
    df.to_csv(path2 + '/' + name2, index=False)

def interpolate_cub(name1 = None, name2 = None):
    df = pd.read_csv(name1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.interpolate(method="cubic")
    df.to_csv(name2, index=False)


def grouped(path1, name1, path2, name2):
    df = pd.read_csv(path1 + '/' + name1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    hour = pd.Timedelta('1h')
    dt = df['Timestamp']
    in_block = (dt.diff() == hour)
    in_block[0] = False
    in_block = in_block.apply(lambda x: not x)
    groups = in_block.cumsum()
    groups = groups.rename("groups")
    df = pd.concat([df, groups], axis=1)
    df.to_csv(path2 + '/' + name2, index=False)

def grouped2(name1, name2):
    df = pd.read_csv(name1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    hour = pd.Timedelta('1h')
    dt = df['Timestamp']
    in_block = (dt.diff() == hour)
    in_block[0] = False
    in_block = in_block.apply(lambda x: not x)
    groups = in_block.cumsum()
    groups = groups.rename("groups")
    df = pd.concat([df, groups], axis=1)
    df.to_csv(name2, index=False)

def TV_data(path_in, name1, path_out, name2):
    df = pd.read_csv(path_in + '/' + name1)
    df = df[['Timestamp', 'Value']]
    df.to_csv(path_out + '/' + name2, index=False)
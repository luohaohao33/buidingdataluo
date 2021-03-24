import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def mean():
    df = pd.read_csv('../train_cuting/train_cutting2_lstm.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    hour = pd.Timedelta('1h')
    dt = df['Timestamp']
    in_block = (dt.diff() == hour)
    in_block[0] = True

    temp_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    temp_mean_imputer.fit(df[['Value']])
    df['Value'] = temp_mean_imputer.transform(df[['Value']])
    df.to_csv('../train_cuting/train_cutting2_lstm_mean.csv', index=False)

if __name__ == '__main__':
    df = pd.read_csv('../train_cuting/train_cutting2_lstm_mean.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    hour = pd.Timedelta('1h')
    dt = df['Timestamp']
    in_block = (dt.diff() == hour)
    in_block[0] = False
    in_block = in_block.apply(lambda x: not x)
    groups = in_block.cumsum()
    groups = groups.rename("groups")
    df = pd.concat([df, groups], axis=1)
    df.to_csv('../train_cuting/train_cutting2_lstm_mean.csv', index=False)











    #print(df.head())

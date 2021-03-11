import pandas as pd
import numpy as np


def add_holidays(df, holiday, n):
    holiday = holiday[holiday['SiteId'] == n]
    holiday['Date'] = pd.to_datetime(holiday['Date'])
    holiday['off1'] = 1
    holiday = holiday.set_index('Date')
    holiday['month'] = holiday.index.month
    holiday['year'] = holiday.index.year
    holiday['mday'] = holiday.index.day
    holiday = holiday.reset_index(level=0)
    holiday = holiday.drop(['Holiday', 'Date'], axis=1)
    df = df.merge(holiday, how='left', on=['month', 'SiteId', 'year', 'mday'])
    df['off2'] = df[['off', 'off1']].max(axis=1)
    df = df.drop(['off', 'off1'], axis=1)
    df.rename(columns={'off2': 'off'}, inplace=True)
    return df

def main():
    holiday = pd.read_csv('../source/holidays.csv')
    for i in range(1, 306):
        train = pd.read_csv('../train_deal_c/train/data_add/data_add{num}.csv'.format(num=i))
        test = pd.read_csv('../test_deal_c/test/data_add/data_add{num}.csv'.format(num=i))
        train = add_holidays(train, holiday, i)
        test = add_holidays(test, holiday, i)
        train.to_csv('../train_deal_c/train/data_add/data_add{num}h.csv'.format(num=i), index=False)
        test.to_csv('../test_deal_c/test/data_add/data_add{num}h.csv'.format(num=i), index=False)
        print(i, flush=True)

if __name__ == '__main__':
    main()







import pandas as pd
import numpy as np
from add_train import process_time , add_weather , add_holiday

def main():
    for n in range (1,306):
        test = pd.read_csv('../test_cuting/test_cutting{num}.csv'.format(num=n), index_col=0)
        weather = pd.read_csv('../weather_cuting/weather_cutting{num}.csv'.format(num=n), index_col=0)
        meta = pd.read_csv('../source/metadata.csv')
        test = process_time(test)
        test = add_weather(test, weather)
        test = add_holiday(test,meta,n)
        test.to_csv('../test_deal_c/test/data_add/data_add{num}.csv'.format(num=n), index=False)


if __name__ == '__main__':
    main()

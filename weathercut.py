import pandas as pd
import numpy as np


weather = pd.read_csv('../source/weather_crt.csv', index_col=0)

def process_num(df):
    for n in range(1 , 306):
        df_temp = df[df['SiteId']==n]
        df_temp.to_csv('../weather_cuting/weather_cutting{num}.csv'.format(num=n))

process_num(weather)
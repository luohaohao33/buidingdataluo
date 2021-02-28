import pandas as pd
import numpy as np


test = pd.read_csv('../source/test.csv', index_col=0)

def process_num(df):
    for n in range(1 , 306):
        df_temp = df[df['SiteId']==n]
        df_temp.to_csv('../test_cuting/test_cutting{num}.csv'.format(num=n))

process_num(test)




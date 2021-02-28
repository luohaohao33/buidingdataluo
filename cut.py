import pandas as pd
import numpy as np


train = pd.read_csv('../source/train.csv', index_col=0)

def process_num(df):
    for n in range(1 , 306):
        df_temp = df[df['SiteId']==n]
        df_temp.to_csv('../train_cuting/train_cutting{num}.csv'.format(num=n))

process_num(train)




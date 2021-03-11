import pandas as pd
import numpy as np

# Imputing missing values in temp and value
from sklearn.impute import SimpleImputer


def process(df):

    if not (pd.isnull(df['Temperature']).all()):

        temp_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        temp_mean_imputer.fit(df[['Temperature']])
        df['Temperature'] = temp_mean_imputer.transform(df[['Temperature']])
        df['Temperature'] = temp_mean_imputer.transform(df[['Temperature']])


    if pd.isnull(df['Value']).all():
        df['Value'] = 0
    else:
        value_mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        value_mean_imputer.fit(df[['Value']])
        df['Value'] = value_mean_imputer.transform(df[['Value']])

    return df

def main():
    for i in range(1, 306):
        train = pd.read_csv('../train_deal_c/train/data_add/data_add{num}h.csv'.format(num=i))
        test = pd.read_csv('../test_deal_c/test/data_add/data_add{num}h.csv'.format(num=i))
        train = process(train)
        test = process(test)
        train.to_csv('../train_deal_c/train/data_repair/data_repair_mean{num}.csv'.format(num=i), index=False)
        test.to_csv('../test_deal_c/test/data_repair/data_repair_mean{num}.csv'.format(num=i), index=False)
        print(i, flush=True)



if __name__ == '__main__':
    main()
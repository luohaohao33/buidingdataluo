
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def line_date_diagam(df,name):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    sns.lineplot(x="Timestamp", y="Value", data=df)
    plt.savefig('../train_deal_c/train/data_repair/%s.png'% name)
    plt.clf()

def main():
    name = 'data_repair_mean1'
    train = pd.read_csv('../train_deal_c/train/data_repair/data_repair_mean1.csv')
    line_date_diagam(train, name)

if __name__ == '__main__':
    plt.style.use("ggplot")
    main()
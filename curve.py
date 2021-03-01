import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
def line_date_diagam(df,name):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    sns.lineplot(x="Timestamp", y="Value", data=df)

    plt.savefig('../line_date/%s.png'% name)
    plt.clf()


for i in range(1,306):
    name = 'line_date'+ str(i)
    site = 'train_cutting'+ str(i)
    train = pd.read_csv('../train_cuting/%s.csv'%site, index_col=0)

    line_date_diagam(train,name)


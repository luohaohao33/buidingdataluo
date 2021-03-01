
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
def box_diagam(df,name):
    #df['Value'] = df['Value'].fillna(0)
    #df['Value_pct'] = df['Value'] / (df['Value'].max()-df['Value'].min())

    sns.boxplot(data=df, y="Value")

    plt.savefig('../test_cuting/%s.png'% name)
    plt.clf()

for i in range(1,306):
    name = 'test_cutting'+ str(i)
    train = pd.read_csv('../test_cuting/%s.csv'%name, index_col=0)

    box_diagam(train,name)
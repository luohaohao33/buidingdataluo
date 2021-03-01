import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
def line_date_diagam(df,name):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    sns.lineplot(x="Timestamp", y="Temperature", data=df)

    plt.savefig('../weather_cuting/%s.png'% name)
    plt.clf()


for i in range(1,306):
    name = 'weather_line_date'+ str(i)
    site = 'weather_cutting'+ str(i)
    weather = pd.read_csv('../weather_cuting/%s.csv'%site, index_col=0)

    line_date_diagam(weather,name)


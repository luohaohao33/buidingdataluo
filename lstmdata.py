import pandas as pd
import numpy as np


df = pd.read_csv('../train_cuting/train_cutting2.csv')


df = df[['Timestamp', 'Value']]
df.to_csv('../train_cuting/train_cutting2_lstm.csv', index=False)


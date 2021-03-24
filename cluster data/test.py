import pandas as pd
from matplotlib import pyplot as plt
import numpy

def miss_3d(path):
    dataset = pd.read_csv(path)
    dataset = dataset.fillna(0)
    
    #print(dataset.isnull().any())





if __name__ == '__main__':
    miss_3d('test_cutting2_lstm.csv')
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def three_D(path):
    dataset = pd.read_csv(path)
    dataset = dataset.interpolate(method="cubic")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    Z = dataset['Value']
    X = dataset['Timestamp'].dt.date
    Y = dataset['Timestamp'].dt.hour
    return X, Y, Z
    # print(dataset.isnull().any())


if __name__ == '__main__':
    X, Y, Z = three_D('test_cutting129V.csv')
    X = dates.date2num(X)
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter(X, Y, Z)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'black'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'green'})
    plt.savefig('test_inter129')
    plt.show()
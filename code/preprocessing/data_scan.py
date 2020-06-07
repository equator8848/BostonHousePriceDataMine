# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 8:35
# @Author  : Equator
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv('../../data/housing.data', names=names, delim_whitespace=True)
    return data


def test_get_data():
    data = get_data()
    print(data.shape)
    print(data.dtypes)
    pd.set_option('display.width', 512)
    pd.set_option('display.max_columns', 16)
    print(data.head(10))
    # pd.set_option('precision', 1)
    print(data.describe())
    print(data.corr(method='pearson'))


def data_split():
    dataset = get_data()
    val = dataset.values
    x = val[:, 0:13]
    y = val[:, 13]
    split_size = 0.2
    seed = 7
    # train_x, test_x, train_y, test_y
    return train_test_split(x, y, test_size=split_size, random_state=seed)


if __name__ == '__main__':
    train_test_split()

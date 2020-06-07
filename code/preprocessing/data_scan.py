# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 8:35
# @Author  : Equator
from pandas import read_csv


def get_data():
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('../../data/housing.data', names=names, delim_whitespace=True)
    return data


def test_get_data():
    data = get_data()
    print(data)


if __name__ == '__main__':
    test_get_data()

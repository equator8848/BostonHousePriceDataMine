# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 9:29
# @Author  : Equator
import matplotlib.pyplot as plt
import code.preprocessing.data_scan as data_scan
from pandas.plotting import scatter_matrix
import numpy as np


# 单一特征图表
def hist(data):
    data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
    plt.show()


# 密度图
def density(data):
    data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, fontsize=1)
    plt.show()


# 箱线图
def box(data):
    data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, fontsize=1)
    plt.show()


# 散点图矩阵
def scatter(data):
    scatter_matrix(data, figsize=(48, 48))
    plt.show()


def figure(data):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    # 刻度
    ticks = np.arange(0, 14, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    names = list(data.columns)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


if __name__ == '__main__':
    data = data_scan.get_data()
    # hist(data)
    # density(data)
    # box(data)
    # scatter(data)
    figure(data)

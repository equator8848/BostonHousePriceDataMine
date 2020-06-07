# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 20:41
# @Author  : Equator
import matplotlib.pyplot as plt


def predicate_rate(predicate_y, reality_y):
    x = [i for i in range(len(predicate_y))]
    l1 = plt.plot(x, predicate_y, 'r', label='predicate')
    l2 = plt.plot(x, reality_y, 'g', label='reality')
    plt.title('Predicate and Reality')
    plt.xlabel('index')
    plt.ylabel('value')
    plt.legend()
    plt.show()

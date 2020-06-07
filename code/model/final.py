# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 20:26
# @Author  : Equator
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from code.preprocessing.data_scan import data_split


def final_model(train_x, train_y, test_x, test_y):
    scaler = StandardScaler().fit(train_x)
    rescaledX = scaler.transform(train_x)
    model = ExtraTreesRegressor(n_estimators=90)
    model.fit(X=rescaledX, y=train_y)
    rescaledX_test = scaler.transform(test_x)
    prediction = model.predict(rescaledX_test)
    print(mean_squared_error(test_y, prediction))


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_split()
    final_model(train_x, train_y, test_x, test_y)

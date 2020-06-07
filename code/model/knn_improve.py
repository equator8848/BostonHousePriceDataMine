# -*- coding: utf-8 -*-
# @Time    : 2020/6/7 17:59
# @Author  : Equator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from code.preprocessing.data_scan import data_split

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


def knn_improve(train_x, train_y):
    scaler = StandardScaler().fit(train_x)
    rescaledX = scaler.transform(train_x)
    param_gird = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
    model = KNeighborsRegressor()
    fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(model, param_gird, scoring=scoring, cv=fold)
    grid_result = grid.fit(X=rescaledX, y=train_y)
    cv_result = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'],
                    grid_result.cv_results_['params'])
    for mean, std, param in cv_result:
        print("%f (%f) with %r" % (mean, std, param))
    print('最优：%s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = data_split()
    knn_improve(train_x, train_y)

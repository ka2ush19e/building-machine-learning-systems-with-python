# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data, target = load_svmlight_file('data/E2006.train')

print('Min target value: {}'.format(target.min()))
print('Max target value: {}'.format(target.max()))
print('Mean target value: {}'.format(target.mean()))
print('Std target value: {}'.format(target.std()))

lr = LinearRegression(fit_intercept=True)
lr.fit(data, target)

pred = lr.predict(data)
print('rmse: {}'.format(np.sqrt(mean_squared_error(target, pred))))

pred = np.zeros_like(target)
kf = KFold(len(target), n_folds=5)
for train, test in kf:
    lr.fit(data[train], target[train])
    pred[test] = lr.predict(data[test])

print('rmse'.format(np.sqrt(mean_squared_error(target, pred))))

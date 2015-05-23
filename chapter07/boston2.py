# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, ElasticNet


boston = load_boston()
x = np.array([np.concatenate((v, [1])) for v in boston.data])
y = boston.target

s, total_error, _, _ = np.linalg.lstsq(x, y)

rmse = np.sqrt(total_error[0] / len(x))
print('rmse: {}'.format(rmse))

x = boston.data
y = boston.target

lr = LinearRegression(fit_intercept=True)
lr.fit(x, y)
p = map(lr.predict, x)
e = p - y
total_error = np.sum(e * e)
print(total_error)
rmse = np.sqrt(total_error / len(p))
print('rmse: {}'.format(rmse))

# en = ElasticNet(fit_intercept=True, alpha=0.5)
# en.fit(x, y)
#
# rmse = np.sqrt(en.residues_/len(x))
# print('rmse: {}'.format(rmse))


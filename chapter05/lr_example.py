# -*- coding: utf-8 -*-

from matplotlib import pyplot
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

np.random.seed(3)

num_per_class = 40
x = np.hstack((norm.rvs(2, size=num_per_class, scale=2), norm.rvs(8, size=num_per_class, scale=3)))
y = np.hstack((np.zeros(num_per_class), np.ones(num_per_class)))


def lr_model(clf, x):
    return 1.0 / (1.0 + np.exp(-(clf.intercept_ + clf.coef_ * x)))


logclf = LogisticRegression()
print(logclf)

logclf.fit(x.reshape(num_per_class * 2, 1), y)
print(np.exp(logclf.intercept_), np.exp(logclf.coef_.ravel()))
print("P(x=-1)=%.2f\tP(x=7)=%.2f" % (lr_model(logclf, -1), lr_model(logclf, 7)))

x_test = np.arange(-5, 20, 0.1)
pyplot.figure(figsize=(10, 4))
pyplot.xlim((-5, 20))
pyplot.scatter(x, y, c=y)
pyplot.plot(x_test, lr_model(logclf, x_test).ravel())
pyplot.plot(x_test, np.ones(x_test.shape[0]) * 0.5, "--")
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.grid(True, linestyle='-', color='0.75')
pyplot.show()



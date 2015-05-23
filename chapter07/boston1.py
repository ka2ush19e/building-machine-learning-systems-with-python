# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_boston


boston = load_boston()
x = boston.data[:, 5]
x = np.array([[v, 1] for v in x])
y = boston.target

s, total_error, _, _ = np.linalg.lstsq(x, y)

rmse = np.sqrt(total_error[0] / len(x))
print('rmse: {}'.format(rmse))

plt.plot(np.dot(x, s), y, 'ro')
plt.plot([0, 50], [0, 50], 'g-')
plt.xlabel('predicted')
plt.ylabel('real')
plt.show()

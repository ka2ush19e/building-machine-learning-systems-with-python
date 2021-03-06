# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names
labels = target_names[target]

is_setosa = (labels == 'setosa')

features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels == 'virginica')

best_acc = -1.0
best_fi = -1.0
best_t = -1.0
for fi in xrange(features.shape[1]):
    thresh = features[:, fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:, fi] > t)
        acc = (pred == is_virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t

print(best_acc, best_fi, best_t)

f0, f1 = 3, 2

area1c = (1., 1, 1)
area2c = (.7, .7, .7)

x0 = features[:, f0].min() * .9
x1 = features[:, f0].max() * 1.1

y0 = features[:, f1].min() * .9
y1 = features[:, f1].max() * 1.1

fig, ax = plt.subplots()
ax.fill_between([best_t, x1], [y0, y0], [y1, y1], color=area2c)
ax.fill_between([x0, best_t], [y0, y0], [y1, y1], color=area1c)
ax.plot([best_t, best_t], [y0, y1], 'k--', lw=2)
ax.scatter(features[is_virginica, f0], features[is_virginica, f1], c='b', marker='o', s=40)
ax.scatter(features[~is_virginica, f0], features[~is_virginica, f1], c='r', marker='x', s=40)
ax.set_ylim(y0, y1)
ax.set_xlim(x0, x1)
ax.set_xlabel(feature_names[f0])
ax.set_ylabel(feature_names[f1])
fig.tight_layout()
plt.show()

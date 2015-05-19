# -*- coding: utf-8 -*-

import json
import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

meta = json.load(open('data/chosen-meta.json'))

questions = [q for q, v in meta.iteritems() if v['ParentId'] == -1]
answers = [q for q, v in meta.iteritems() if v['ParentId'] != -1]

y = np.asarray([meta[aid]['Score'] > 0 for aid in answers])
x = np.asarray([[meta[aid]['LinkCount'], meta[aid]['NumCodeLines'], meta[aid]['NumTextTokens']] for aid in answers])

scores = []
precisions, recalls, thresholds = [], [], []

cv = KFold(n=len(x), n_folds=10, indices=True)

for train, test in cv:
    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]

    # clf = neighbors.KNeighborsClassifier()
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    scores.append(clf.score(x_test, y_test))

    precision, recall, pr_thresholds = precision_recall_curve(y_test, clf.predict(x_test))
    print(precision, recall, pr_thresholds)
    print(clf.coef_)

print(np.mean(scores), np.std(scores))



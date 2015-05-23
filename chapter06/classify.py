# -*- coding: utf-8 -*-

from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import utils


def create_model():
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', vectorizer), ('clf', clf)])
    return pipeline


def train(clf_factory, X, Y):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)
    scores = []
    pr_scores = []
    for train, test in cv:
        x_train, y_train = X[train], Y[train]
        x_test, y_test = X[test], Y[test]

        clf = clf_factory()
        clf.fit(x_train, y_train)

        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)

        scores.append(test_score)
        proba = clf.predict_proba(x_test)

        precision, recall, pr_thresholds = precision_recall_curve(y_test, proba[:, 1])
        pr_scores.append(auc(recall, precision))

    print(np.mean(scores), np.std(scores), np.mean(pr_scores), np.std(pr_scores))


def tweak_labels(Y, pos_sent_list):
    pos = Y == pos_sent_list[0]
    for sent_label in pos_sent_list[1:]:
        pos |= Y == sent_label

    Y = np.zeros(Y.shape[0])
    Y[pos] = 1
    Y = Y.astype(int)

    return Y


def grid_search_model(clf_factory, X, Y):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, random_state=0)

    param_grid = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      vect__min_df=[1, 2],
                      vect__stop_words=[None, "english"],
                      vect__smooth_idf=[False, True],
                      vect__use_idf=[False, True],
                      vect__sublinear_tf=[False, True],
                      vect__binary=[False, True],
                      clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],
                      )

    grid_search = GridSearchCV(
        clf_factory(), param_grid=param_grid, cv=cv, score_func=f1_score, verbose=10)
    grid_search.fit(X, Y)
    clf = grid_search.best_estimator_
    print(clf)

    return clf


X_org, Y_org = utils.load()
classes = np.unique(Y_org)
for c in classes:
    print('{}: {}'.format(c, sum(Y_org == c)))

pos_neg_idx = np.logical_or(Y_org == 'positive', Y_org == 'negative')
X = X_org[pos_neg_idx]
Y = Y_org[pos_neg_idx]
Y = tweak_labels(Y, ["positive"])
train(create_model, X, Y)

X = X_org
Y = tweak_labels(Y_org, ["positive", 'negative'])
train(create_model, X, Y)

X = X_org
Y = tweak_labels(Y_org, ["positive"])
train(create_model, X, Y)

X = X_org
Y = tweak_labels(Y_org, ["positive", 'negative'])
grid_search_model(create_model, X, Y)

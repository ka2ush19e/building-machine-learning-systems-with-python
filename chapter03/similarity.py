# -*- coding: utf-8 -*-

import math
import os
import sys

from nltk.stem import SnowballStemmer
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    posts = [open(os.path.join('data', f)).read() for f in os.listdir('data')]

    stemmer = SnowballStemmer('english')

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

    vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english')

    x_train = vectorizer.fit_transform(posts)
    num_samples, num_features = x_train.shape
    print('samples: {}, features: {}'.format(num_samples, num_features))
    print

    new_post_vec = vectorizer.transform(['imaging databases'])
    print(new_post_vec)
    print

    print(new_post_vec.toarray())
    print

    dist_func = dist_norm
    best_doc = None
    best_dist = sys.maxint
    best_i = None
    for i, post in enumerate(posts):
        post_vec = x_train.getrow(i)
        dist = dist_func(post_vec, new_post_vec)
        print('=== Post {} with dist={}: {}'.format(i, dist, post))
        if dist < best_dist:
            best_doc = post
            best_dist = dist
            best_i = i
    print('Best Post is {} with dist={}: {}'.format(best_i, best_dist, best_doc))


def dist_raw(x, y):
    delta = x - y
    return sp.linalg.norm(delta.toarray())


def dist_norm(x, y):
    x_normalized = x / sp.linalg.norm(x.toarray())
    y_normalized = y / sp.linalg.norm(y.toarray())
    delta = x_normalized - y_normalized
    return sp.linalg.norm(delta.toarray())


if __name__ == '__main__':
    main()

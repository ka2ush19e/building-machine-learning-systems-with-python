# -*- coding: utf-8 -*-

from gensim import corpora, matutils, models, similarities
import numpy as np
from scipy.spatial import distance

corpus = corpora.BleiCorpus('data/ap/ap.dat', 'data/ap/vocab.txt')

model = models.LdaModel(corpus, 100, corpus.id2word)

topics = [model[c] for c in corpus]
print('topic sample: {}'.format(topics[0]))

dense = np.zeros((len(topics), 100), float)
for ti, t in enumerate(topics):
    for tj, v in t:
        dense[ti, tj] = v

pairwise = distance.squareform(distance.pdist(dense))

largest = pairwise.max() + 1
for i in range(len(topics)):
    pairwise[i, i] = largest

print(pairwise[1].argmin())

# -*- coding: utf-8 -*-

import logging
import os
import gensim
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

id2word = gensim.corpora.Dictionary.load_from_text('data/wiki/wiki_en_output_wordids.txt.bz2')
mm = gensim.corpora.MmCorpus('data/wiki/wiki_en_output_tfidf.mm')

if os.path.exists('wiki_lda.pkl'):
    model = gensim.models.LdaModel.load('wiki_lda.pkl')
else:
    model = gensim.models.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1,
                                   chunksize=10000, passes=1)
    model.save('wiki_lda.pkl')

topics = [model[doc] for doc in mm]
print(topics[0])

lens = np.array([len(t) for t in topics])
print(np.mean(lens))
print(np.mean(lens <= 10))

counts = np.zeros(100)
for doc_top in topics:
    for ti, _ in doc_top:
        counts[ti] += 1
words = model.show_topic(counts.argmax(), 64)

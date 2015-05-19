# -*- coding: utf-8 -*-

from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp


def main():
    # all_data = fetch_20newsgroups(subset="all")
    # print("total posts: {}".format(len(all_data)))

    groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
    train_data = fetch_20newsgroups(subset='train', categories=groups)
    print("train posts: {}".format(len(train_data.filenames)))

    stemmer = SnowballStemmer('english')

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

    vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english',
                                        decode_error='ignore')
    vectorized = vectorizer.fit_transform(train_data.data)
    sample_count, feature_count = vectorized.shape
    print('samples: {}, features: {}'.format(sample_count, feature_count))

    num_clusters = 50
    km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
    km.fit(vectorized)
    print('km.labels_: {}'.format(km.labels_))

    new_post = """Disk drive problems. Hi, I have a problem with my hard disk.
    After 1 year it is working only sporadically now.
    I tried to format it, but now it doesn't boot any more.
    Any ideas? Thanks.
    """
    new_post_vector = vectorizer.transform([new_post])
    new_post_label = km.predict(new_post_vector)[0]

    similar_indices = (km.labels_ == new_post_label).nonzero()[0]
    similar = []
    for i in similar_indices:
        dist = sp.linalg.norm((new_post_vector - vectorized[i]).toarray())
        similar.append((dist, train_data.data[i]))
    similar = sorted(similar)
    print('similar docs: {}'.format(len(similar)))

    show_at_1 = similar[0]
    show_at_2 = similar[len(similar) / 2]
    show_at_3 = similar[-1]

    print('### 1 ###')
    print(show_at_1)
    print

    print('### 2 ###')
    print(show_at_2)
    print

    print('### 3 ###')
    print(show_at_3)
    print


if __name__ == '__main__':
    main()

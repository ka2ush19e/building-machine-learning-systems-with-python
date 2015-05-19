# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)

print(vectorizer)
print

X = vectorizer.fit_transform([
    'How to format my hard disk',
    'Hard disk format problems'
])
print(vectorizer.get_feature_names())
print

print(X.toarray().transpose())

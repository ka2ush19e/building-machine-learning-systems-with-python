# -*- coding: utf-8 -*-

import csv
import json
import os
import numpy as np


def load(line_count=-1):
    count = 0

    topics = []
    labels = []
    tweets = []

    with open(os.path.join('data', "corpus.csv"), "r") as csvfile:
        metareader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in metareader:
            count += 1
            if 0 < line_count < count:
                break

            topic, label, tweet_id = line

            tweet_fn = os.path.join('data', 'rawdata', '%s.json' % tweet_id)
            try:
                tweet = json.load(open(tweet_fn, "r"))
            except IOError:
                # print("Tweet '%s' not found. Skip." % tweet_fn)
                continue

            if 'text' in tweet and tweet['user']['lang'] == "en":
                topics.append(topic)
                labels.append(label)
                tweets.append(tweet['text'])

    tweets = np.asarray(tweets)
    labels = np.asarray(labels)

    return tweets, labels

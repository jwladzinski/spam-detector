# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import pandas as pd
from steem import Steem
from steem.blockchain import Blockchain
from steem.post import Post
from steem.blog import Blog
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# private posting key from environment variable
POSTING_KEY = os.getenv('POSTING_KEY')

# Multinomial Naive Bayes based spam filter trained from input file
class NaiveBayesSpamFilter:
    def __init__(self, training_file):
        # read data from training file, each row contains label and message separated by '\t'
        self.messages = pd.read_csv(training_file, sep='\t', quoting=csv.QUOTE_NONE, names=['label', 'message'])
        self.bag_of_words_transformer = CountVectorizer(analyzer=self.split_into_lemmas).fit(self.messages['message'])
        self.messages_bag_of_words = self.bag_of_words_transformer.transform(self.messages['message'])
        self.tfidf_transformer = TfidfTransformer().fit(self.messages_bag_of_words)
        self.messages_tfidf = self.tfidf_transformer.transform(self.messages_bag_of_words)
        # train Multinomial Naive Bayes algorithm with training data
        self.multinomial_nb = MultinomialNB().fit(self.messages_tfidf, self.messages['label'])
      
    # return probability that given message is spam (0.0 - 1.0)
    def spam_probability(self, message):
        bag_of_words = self.bag_of_words_transformer.transform([message])
        tfidf = self.tfidf_transformer.transform(bag_of_words)
        return self.multinomial_nb.predict_proba(tfidf)[0][1]
  
    # split text into list of lemmas
    # 'Apples and oranges' -> ['apple', 'and', 'orange']
    def split_into_lemmas(self, message):
        return [word.lemma for word in TextBlob(message.lower()).words]

class SpamDetectorBot:
    def __init__(self, config, model):
        # retrieve parameters from config file
        self.account = config['account']
        self.nodes = config['nodes']
        self.tags = config['tags']
        self.probability_threshold = config['probability_threshold']
        self.training_file = config['training_file']
        self.reply_mode = config['reply_mode']
        self.vote_mode = config['vote_mode']
        self.vote_weight = config['vote_weight']

        self.steem = Steem(nodes=self.nodes, keys=[POSTING_KEY])

        # machine learning model (=algorithm)
        self.model = model

        # comments that was previously seen (every edit of comment is de facto creating new comment
        # a we don't want to analyze one comment multiple times)
        self.seen = set()

def main():
    # read config file
    config = json.loads(open(sys.argv[1]).read())
    # create model
    model = NaiveBayesSpamFilter(config['training_file'])
    # create bot
    bot = SpamDetectorBot(config, model)

if __name__ == '__main__':
    main()
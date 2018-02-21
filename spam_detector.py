# -*- coding: utf-8 -*-

import sys
import json
import csv
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

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

def main():
    # read config file
    config = json.loads(open(sys.argv[1]).read())
    # create model
    model = NaiveBayesSpamFilter(config['training_file'])

    print(model.spam_probability('Upvote and resteem!'))


if __name__ == '__main__':
    main()
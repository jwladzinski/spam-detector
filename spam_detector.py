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
from steembase.exceptions import PostDoesNotExist
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup

# private posting key from environment variable
POSTING_KEY = os.getenv('POSTING_KEY')

# removes markdown and html tags from text
def remove_html_and_markdown(text):
    return ''.join(BeautifulSoup(text, 'lxml').findAll(text=True))

def replace_white_spaces_with_space(text):
    return text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

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

    # returns main post from given comment
    def main_post(self, post):
        return Post(post['url'].split('#')[0], steemd_instance=self.steem)

    # log to console
    def log(self, p, author, message):
        print('Spam score: %.2f%%  |  ' % (100 * p), '@' + author + ':', message[:50])

    # every comment that is classified as spam, is added to training file
    def append_message(self, label, message):
        with open(self.training_file, 'a') as f:
            f.write(label + '\t' + replace_white_spaces_with_space(message) + '\n')

    # response that will be written by bot
    def response(self, p):
        return 'Please stop spamming in comments or else people may flag you!\n<sup>Spam probability: %.2f%% </sup>' % (100 * p)

    def run(self):
        blockchain = Blockchain(steemd_instance=self.steem)
        # stream of comments
        stream = blockchain.stream(filter_by=['comment'])
        while True:
            try:
                for comment in stream:
                    post = Post(comment, steemd_instance=self.steem)
                    if not post.is_main_post() and post['url'] not in self.seen:
                        main_post = self.main_post(post)
                        # if self.tags is empty bot analyzes all tags
                        # otherwise bot analyzes only comments that contains at least one of given tag          
                        if not self.tags or (set(self.tags) & set(main_post['tags'])):      
                            message = remove_html_and_markdown(post['body'].strip()) 
                            message = replace_white_spaces_with_space(message)
                            # calculates probability that given message is spam
                            p = self.model.spam_probability(message)
                            self.log(p, post['author'], message)
                            self.seen.add(post['url'])
                            # if probability is greater than threshold
                            if p > self.probability_threshold:
                                self.append_message('spam', message)
                                response = self.response(p)
                                if self.reply_mode:
                                    post.reply(response, '', self.account)
                                if self.vote_mode:
                                    post.upvote(weight=self.vote_weight, voter=self.account)    
            except PostDoesNotExist as pex:
                continue
            except Exception as ex:
                print(repr(ex))
                continue

def main():
    # read config file
    config = json.loads(open(sys.argv[1]).read())
    # create model
    model = NaiveBayesSpamFilter(config['training_file'])
    # create bot
    bot = SpamDetectorBot(config, model)
    # start bot
    bot.run()

if __name__ == '__main__':
    main()
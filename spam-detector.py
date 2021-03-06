# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from pprint import pprint
from collections import Counter
import itertools
from operator import itemgetter
from steem import Steem
from steem.blockchain import Blockchain
from steem.steemd import Steemd
from steem.instance import set_shared_steemd_instance
from steem.post import Post
from steem.blog import Blog
from steembase.exceptions import PostDoesNotExist
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

steemd_nodes = [
    'https://api.steemit.com/',
    'https://gtg.steem.house:8090/',
    'https://steemd.steemitstage.com/',
    'https://steemd.steemgigs.org/'
]
set_shared_steemd_instance(Steemd(nodes=steemd_nodes))

# private posting key from environment variable
POSTING_KEY = os.getenv('POMOCNIK_POSTING_KEY')

# removes markdown and html tags from text
def remove_html_and_markdown(text):
    return ''.join(BeautifulSoup(text, 'lxml').findAll(text=True))

def replace_white_spaces_with_space(text):
    return text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

def get_message_from_post(post):
    message = post['body'].strip() 
    return replace_white_spaces_with_space(remove_html_and_markdown(message))

class StackedModel():
    def __init__(self, models, X_train, y_train, X_test, y_test):
        self.models = models
        self.models = [model.fit(X_train, y_train) for model in self.models]

        y_predicts = [model.predict(X_test) for model in self.models]
        for y_predict, model in zip(y_predicts, self.models):
            print(model, '\n')
            print('Confusion matrix:')
            print(confusion_matrix(y_test, y_predict), '\n')

        print('Stacked model')
        print('Confusion matrix:')    
        stacked_y_predict = [self.predict(x) for x in X_test]

        stacked_confusion_matrix = confusion_matrix(y_test, stacked_y_predict)
        print(stacked_confusion_matrix, '\n')
        self.plot_confusion_matrix(stacked_confusion_matrix , classes=['Spam', 'Ham'])
        
    def predict(self, x):
        return 'spam' if self.predict_proba(x) >= 0.5 else 'ham'

    def predict_proba(self, x):
        probas = [model.predict_proba(x)[0][1] for model in self.models]
        weighted_proba = sum(probas) / len(probas)
        return weighted_proba

    def plot_confusion_matrix(self, confusion_matrix, classes):

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        threshold = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(2), range(2)):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > threshold else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig.savefig('fig.png')

class SpamFilter:
    def __init__(self, training_file):
        # read data from training file, each row contains label and message separated by '\t'
        self.messages = pd.read_csv(training_file, sep='\t', quoting=csv.QUOTE_NONE, names=['label', 'message'])

        X = self.messages['message']
        y = self.messages['label']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.bag_of_words_transformer = CountVectorizer(
            analyzer=self.split_into_lemmas,
            stop_words='english',
            strip_accents='unicode').fit(self.X_train)

        self.messages_bag_of_words = self.bag_of_words_transformer.transform(self.X_train)
        self.tfidf_transformer = TfidfTransformer().fit(self.messages_bag_of_words)
        self.X_train = self.tfidf_transformer.transform(self.messages_bag_of_words)
        self.X_test = self.to_tfidf(self.X_test)

        C = 1.0
        # 4 algorithms stacked in one
        self.model = StackedModel([
            MultinomialNB(),
            SVC(kernel='linear', C=C, probability=True),
            SVC(kernel='rbf', gamma=0.7, C=C, probability=True),
            NuSVC(probability=True)
            ], 
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test) 
    
    def to_tfidf(self, X):
        bag_of_words = self.bag_of_words_transformer.transform(X)
        tfidf = self.tfidf_transformer.transform(bag_of_words)
        return tfidf

    # return probability that given message is spam (0.0 - 1.0)
    def spam_probability(self, X):
        tfidf = self.to_tfidf([X])
        return self.model.predict_proba(tfidf)

    def average_spam_probability(self, blog, k):
        previous_messages = [get_message_from_post(previous_post) for previous_post in blog.take(k)]
        counter = Counter()
        for message in previous_messages:
            counter[message] += 1

        most_common = counter.most_common(1)[0]
        generic_comment = None
        rep = most_common[1]

        # 50% comments are the repeated
        if rep >= 0.5 * k:
            generic_comment = most_common[0]

        scores = [self.spam_probability(m) for m in previous_messages]
        average = sum(scores) / len(scores)
        return average, generic_comment, rep
  
    # split text into list of lemmas
    # 'Apples and oranges' -> ['apple', 'and', 'orange']
    def split_into_lemmas(self, message):
        lemmas = [word.lemma for word in TextBlob(message.lower()).words]
        lemmas = filter(lambda w: w.isalpha(), lemmas)
        lemmas = filter(lambda w: len(w) > 1, lemmas)
        return lemmas

class SpamDetectorBot:
    def __init__(self, config, model):
        # retrieve parameters from config file
        self.account = config['account']
        self.nodes = config['nodes']
        self.tags = config['tags']
        self.probability_threshold = config['probability_threshold']
        self.training_file = config['training_file']
        self.blacklist_file = config['blacklist_file']
        self.whitelist_file = config['whitelist_file']
        self.scamlist_file = config['scamlist_file']
        self.reply_mode = config['reply_mode']
        self.vote_mode = config['vote_mode']
        self.vote_weight = config['vote_weight']
        self.blacklist = [user.strip() for user in open(self.blacklist_file, 'r').readlines()]
        self.whitelist = [user.strip() for user in open(self.whitelist_file, 'r').readlines()]
        self.scamlist = [user.strip() for user in open(self.scamlist_file, 'r').readlines()]
        self.num_previous_comments = config['num_previous_comments']
        self.steem = Steem(nodes=self.nodes, keys=[POSTING_KEY])

        # machine learning model (=algorithm)
        self.model = model

    # returns main post from given comment
    def main_post(self, post):
        return Post(post['url'].split('#')[0], steemd_instance=self.steem)

    # log to console
    def log(self, spam_probabilty, average_spam_probability, author, message):
        print('Spam score: %4.2f%% / %4.2f%% |  ' % (100 * spam_probabilty, 100 * average_spam_probability),
         '@' + author + ':', message[:50])

    # every comment that is classified as spam, is added to training file
    def append_message(self, label, message):
        with open(self.training_file, 'a') as f:
            f.write(label + '\t' + replace_white_spaces_with_space(message) + '\n')

    # response that will be written by bot
    def response(self, p, generic_comment, rep):

        generic_part = ('Last %d/%d of your comments have content:\n> %s\n' % (rep, self.num_previous_comments, generic_comment)) if generic_comment else ''
        return (
        'Please stop spamming in comments or else people will flag you!\n' + 
        generic_part +
        '\n<sup>Spam probability: %.2f%% </sup>' % (100 * p))

    def reply(self, post, response):
        if self.account in [p['author'] for p in post.get_replies()]:
            return
        try:
            post.reply(response, '', self.account)
        except Exception as ex:
            print(ex)

    def vote(self, post):
        if self.account in [v['voter'] for v in post['active_votes']]:
            return
        try:
            post.upvote(weight=self.vote_weight, voter=self.account)
        except Exception as ex:
            print(ex)

    def append_to_blacklist(self, user):
        if user not in self.blacklist:
            self.blacklist.append(user)
            with open(self.blacklist_file, 'a') as f:
                f.write(user + '\n')

    def filter_by_tag(self, tags):
        return not self.tags or (set(self.tags) & set(tags))

    def run(self):
        blockchain = Blockchain(steemd_instance=self.steem)
        # stream of comments
        stream = blockchain.stream(filter_by=['comment'])
        while True:
            try:
                for comment in stream:
                    post = Post(comment, steemd_instance=self.steem)
                    if not post.is_main_post() and post['url']:
                        main_post = self.main_post(post)
                        # if self.tags is empty bot analyzes all tags
                        # otherwise bot analyzes only comments that contains at least one of given tag          
                        if self.filter_by_tag(main_post['tags']):

                            # users on whitelist are ignored
                            if post['author'] in self.whitelist:
                                print('Ignored:', post['author'])
                                continue

                            # users on scamlist are flagged without immediately
                            elif post['author'] in self.scamlist:
                                print('Scam:', post['author'])
                                self.reply(post, 'Scam alert! Do not click in link!')
                                self.vote(post)
                                
                            else:    
                                message = get_message_from_post(post)
                                spam_probability = self.model.spam_probability(message) 
                                blog = Blog(account_name=post['author'], comments_only=True, steemd_instance=self.steem)
                                average_spam_probability, generic_comment, rep = self.model.average_spam_probability(blog, self.num_previous_comments)
                                print('*' if average_spam_probability > self.probability_threshold else ' ', end='')       
                                self.log(spam_probability, average_spam_probability, post['author'], message)

                                if spam_probability > self.probability_threshold and average_spam_probability > self.probability_threshold:
                                    self.append_to_blacklist(post['author'])
                                    self.append_message('spam', message)
                                    response = self.response(average_spam_probability, generic_comment, rep)
                                    inp = input('y/n\n') # currently I approve manually 
                                    if inp == 'y':
                                        if self.reply_mode:
                                            self.reply(post, response)
                                        if self.vote_mode:
                                            self.vote(post)
            except PostDoesNotExist as pex:
                continue
            except Exception as ex:
                print(repr(ex))
                continue

def main():
    # read config file
    config = json.loads(open(sys.argv[1]).read())
    # create model
    model = SpamFilter(config['training_file'])
    # create bot
    bot = SpamDetectorBot(config, model)
    # start bot
    bot.run()

if __name__ == '__main__':
    main()
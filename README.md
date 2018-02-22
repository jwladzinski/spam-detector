### spam-detector

spam-detector is a bot designed to detect spam comments on Steem blockchain. It uses Multinomial Naive Bayes algorithm. It can reply to spam comment or downvote it.

### Requirements
- python3.6
- libraries: steem-python, scikit-learn, pandas, textblob, bs4

Repository contains requirements.txt file.

### Running 

`$ POSTING_KEY=<posting_key> spam_detector config.json`

Private posting key is stored as environent variable.

### Configuration

All parameters are stored in config.json file.

Key | Value
-|-
account | account used by bot
nodes | list of Steem nodes
tags | tags which are observed
probability_threshold | threshold to classify as spam
training_file | input training file
reply_mode | 0 - without reply, 1 - with reply
vote_mode | 0 - without vote, 1 with vote
vote_weight | weight of the vote from range [-100.0, 100.0]
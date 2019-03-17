## Might need to train as 2500 tweets will only be collected due to rate limits

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import os, json, re, sys, csv
import pandas as pd
import numpy as np

import datetime

class getHistorical(object):

    def __init__(self, historical_file):
        self.historical_tweets_file = historical

    def getTweets(self):
        pass
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import os, json, re, sys, csv
import pandas as pd
import numpy as np
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import datetime
now = datetime.datetime.now()

from tqdm import tqdm

import spam_filter

# Interface connetions
#import zerorpc

# Provides list of unicode emojis for extraction
import emoji as ji
 
from dotenv import load_dotenv
from pathlib import Path  # python3 only
env_path = Path('.') / 'data_collector/twitter/config/twitter.env'
load_dotenv(dotenv_path=env_path)

class keys():

    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.access_secret = os.getenv("ACCESS_SECRET")
        self.currency_hashtags = os.getenv("CURRENCY_HASHTAGS")

class utilityFuncs():

    def __init__(self):
        pass

    def cleanTweet(self, text):
        # Function to clean tweets, removes links and special characters
        return re.sub(r'([^0-9A-Za-z \-\%\£\$ \t])|(@[A-Za-z0-9]+)|(http\S+)', '', text), ' '.join(c for c in text if c in ji.UNICODE_EMOJI)
    
    def removeSpacing(self, text):
        return re.sub(r'( +)', ' ', text)

    def fixLines(self, text):
        return re.sub(r"([\r\n])", " ", text)
        #return re.sub(r"(\w)([A-Z])", r"\1 \2", text)


    def remove_non_ascii(self, text):
        return ''.join(i for i in text if ord(i)<128)
    
    def detectLaguage(self, text):
        """
        Calculate the probability of given text is written in several languages
        Using nltk stopwords and comparing to all supported languages

        There are other ways to identify this - TextBlob.detect_language and Ngrams
        """

        language_ratios = {}
        tokens = wordpunct_tokenize(text)
        words = [word.lower() for word in tokens]

        # Compute per language in nltk number of stopwords in text
        for language in stopwords.fileids():
            stopwords_set = set(stopwords.words(language))
            words_set = set(words)
            common_elements = words_set.intersection(stopwords_set)
        
            language_ratios[language] = len(common_elements) # Ratio scores

        ratios = language_ratios

        ##print(ratios)
        highest_ratio = max(ratios, key=ratios.get)

        print("Console: Text is - ", highest_ratio)
        sys.stdout.flush()

        if highest_ratio == 'english':
            return True
        else:
            return False

    def checkLength(self, text):
        tokens = text.split()
        if len(tokens) <= 5:
            return False
        else:
            return True

#class spamFiltering():
#    """
#    Spam filter using a naive bayes classifier 
#    TODO: Will need to generate a training dataset both spam and accepted messages
#    """
#    
#    def __init__(self, training_set):
#        self.training_set = training_set#
#
#    def trainFilter(self):
#        dataset = spamFiltering().loadDataset(self.training_set)
#        #ham = loadDatasets('PATH')
#
#        all_text = [(text, 'text') for text in dataset]
#
#        # Randomise/shuffle dataset
#        random.shuffle(all_text)
#        
#    def loadDataset(self, path):
#        list = []
#        with open(path) as file:
#            tweet_data = json.load(file)
#            list.append()
#
#        return False

class Streamer():

    def __init__(self):
        pass

    def stream_tweets(self, tweets_file, training_set, hashtag):
        listener = Listener(tweets_file, training_set)
        auth = OAuthHandler(keys().api_key, keys().api_secret)

        print("Console: ", "Authorising with twitter API")
        sys.stdout.flush()

        auth.set_access_token(keys().access_token, keys().access_secret)

        print("Console: ", "Streaming Tweets")
        stream = Stream(auth, listener, tweet_mode='extended')
        stream.filter(languages=["en"], track=hashtag)

class Listener(StreamListener):
    
    def __init__(self, tweets_file, training_set):
        self.tweets_file = tweets_file
        self.training_set = training_set
    
    def on_data(self, data):

        data = json.loads(data)
        
        try:
            # Check if tweet is a retweet
            if 'retweeted_status' in data:
                if 'extended_tweet' in data['retweeted_status']:
                    #if tweet is over the 140 word limit
                    text = data['retweeted_status']['extended_tweet']['full_text']
                    print("Uncleaned Tweet:", text)
                    sys.stdout.flush()
                else:
                    text = data['retweeted_status']['text']
                    print("Uncleaned Tweet:", text)
                    sys.stdout.flush()
            else:
                # Else if a normal Tweeet
                if 'extended_tweet' in data:
                    # If tweet is over 140 word limit
                    text = data['extended_tweet']['full_text']
                    print("Uncleaned Tweet:", text)
                    sys.stdout.flush()
                else:
                    text = data['text']
                    print("Uncleaned Tweet: ", text)
                    sys.stdout.flush()

            removedLines = utilityFuncs().fixLines(text)
            removedSpecialChars = utilityFuncs().cleanTweet(removedLines)
            removedSpacing = utilityFuncs().removeSpacing(removedSpecialChars[0])

            tweetLength = utilityFuncs().checkLength(removedSpacing)


            if tweetLength == True:

                checkIfEnglish = utilityFuncs().detectLaguage(removedSpecialChars[0])


                if checkIfEnglish == True:

                    tweetText = utilityFuncs().remove_non_ascii(removedSpacing)

                    print("Cleaned Tweet: ", tweetText)
                    sys.stdout.flush()

                    cleanedTweet = tweetText+' '+removedSpecialChars[1]

                    try:
                        with open(tweets_file, mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                            writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:%M"), 'tweet': cleanedTweet})

                        return True
                    except BaseException as exception:
                        print("Error: %s" % str(exception))
                        sys.stdout.flush()
                    return False

                ## Create spam training set
                #try:
                #    list = []
                #    with open(self.training_set) as file:
                #        for line in file:
                #            list.append(line)
                #    list.append(tweet)
                #    with open(self.training_set, 'w') as file:
                #        file.write(list)
                #except BaseException as e:
                #    print("Error: %s" % str(e))
                else:
                    print("Console: ", "Dropping tweet as it is not English")
                    sys.stdout.flush()
            else:
                print("Console: ", "Tweet too short for analysis")
                sys.stdout.flush()
        except BaseException as e:
                print("Console: ", "Error: %s" % str(e))
                sys.stdout.flush()
        return True
          
    def on_error(self, status_code):
        if status_code == 420:
            return False
        print("Console: ", status_code)
        sys.stdout.flush()

 
if __name__ == '__main__':
 
    print("Console: ", "==== tweet_collector.py ====")
    sys.stdout.flush()

    hashtag = keys().currency_hashtags
    hashtag = hashtag.split(', ')
    tweets_file = "data_collector/tweets.csv"
    training_set = "data_collector/500_spam_ham.csv"
    tweet_data = []

    print("Console:", "Initialising CSV...")
    sys.stdout.flush()


    with open(tweets_file, mode='w') as csv_file:
        fieldnames = ['created_at', 'tweet']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

    print("Console:", "Training Spam Filter...")
    data = pd.read_csv(training_set)

    data['class'] = data['classes'].map({'ham': 0, 'spam': 1})

    data.drop(['classes'], axis=1, inplace=True)
    
    trainIndex, testIndex = list(), list()
    for i in tqdm(range(data.shape[0])):
        if np.random.uniform(0, 1) < 0.75:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = data.loc[trainIndex]
    testData  = data.loc[testIndex]

    trainData.reset_index(inplace=True)
    testData.reset_index(inplace=True)
    trainData.drop(['index'], axis=1, inplace=True)
    testData.drop(['index'], axis=1, inplace=True)

    print("TRAIN DATA", trainData)
    print("TEST DATA: ", testData['tweet'])

    spamFilter = spam_filter.classifier(trainData)
    spamFilter.train()
    prediction = spamFilter.predict(testData['tweet'])

    print("PREDICTION: ", prediction)

    boop = spamFilter.predict("Bitcoin SV toughens up its cryptocurrency")

    print("BOOP: ", boop)

    print("Console:", "Starting Twitter Streamer")
    sys.stdout.flush()
    
    #twitter_streamer = Streamer()
    #twitter_streamer.stream_tweets(tweets_file, hashtag)
    
    
    #spamFiltering(training_set)
    
    
    #addr = 'tcp://127.0.0.1:8686'
    #server = zerorpc.Server(twitter_streamer.stream_tweets(tweets_file, training_set, hashtag))
    #server.bind(addr)
    #print("Process running on {}".format(addr))
    #server.run()
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import os, json, re, sys
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import random

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
        return re.sub(r'([^0-9A-Za-z \t])|(@[A-Za-z0-9]+)|(http\S+)', '', text), ' '.join(c for c in text if c in ji.UNICODE_EMOJI)
    
    def removeSpacing(self, text):
        return re.sub(r'( +)', ' ', text)

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
                    print("Uncleaned Tweet", text)
                    sys.stdout.flush()
                else:
                    text = data['retweeted_status']['text']
                    print("Uncleaned Tweet", text)
                    sys.stdout.flush()
            else:
                # Else if a normal Tweeet
                if 'extended_tweet' in data:
                    # If tweet is over 140 word limit
                    text = data['extended_tweet']['full_text']
                    print("Uncleaned Tweet", text)
                    sys.stdout.flush()
                else:
                    text = data['text']
                    print("Uncleaned Tweet: ", text)
                    sys.stdout.flush()
            
            tweet = utilityFuncs().cleanTweet(text)
            tweetText = utilityFuncs().removeSpacing(tweet[0])

            checkIfEnglish = utilityFuncs().detectLaguage(tweet[0])

            if checkIfEnglish == True:

                tweetText = utilityFuncs().remove_non_ascii(tweetText)

                print("Cleaned Tweet: ", tweetText)
                sys.stdout.flush()

                tweet = tweetText+' '+tweet[1]

                try:
                    with open(self.tweets_file) as file:
                        tweet_data = json.load(file)
                    tweet_data.append({
                        'created_at'    : data['created_at'],
                        'text'          : tweet,
                        'reply_count'   : data['reply_count'],
                        'retweet_count' : data['retweet_count'],
                        'favorite_count': data['favorite_count']
                    })

                    with open(self.tweets_file, 'w') as file:
                        json.dump(tweet_data, file, sort_keys=True, indent=4)
                    return True
                except BaseException as exception:
                    print("Error: %s" % str(exception))
                return True

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
    tweets_file = "data_collector/tweets.json"
    training_set = "data_collector/training_set.txt"
    tweet_data = []

    print("Console:", "Starting Twitter Streamer")
    sys.stdout.flush()
    
    twitter_streamer = Streamer()
    twitter_streamer.stream_tweets(tweets_file, training_set, hashtag)
    #spamFiltering(training_set)
    #addr = 'tcp://127.0.0.1:8686'
    #server = zerorpc.Server(twitter_streamer.stream_tweets(tweets_file, training_set, hashtag))
    #server.bind(addr)
    #print("Process running on {}".format(addr))
    #server.run()
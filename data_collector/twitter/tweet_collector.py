from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import os, json, re

# Provides list of unicode emojis for extraction
import emoji as ji
 
from dotenv import load_dotenv
from pathlib import Path  # python3 only
env_path = Path('.') / 'config/twitter.env'
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

class Streamer():

    def __init__(self):
        pass

    def stream_tweets(self, tweets_file, hashtag):
        listener = Listener(tweets_file)
        auth = OAuthHandler(keys().api_key, keys().api_secret)
        auth.set_access_token(keys().access_token, keys().access_secret)
        stream = Stream(auth, listener, tweet_mode='extended')
        stream.filter(languages=["en"], track=hashtag)

class Listener(StreamListener):
    
    def __init__(self, tweets_file):
        self.tweets_file = tweets_file
    
    def on_data(self, data):

        data = json.loads(data)

        try:
            # Check if tweet is a retweet
            if 'retweeted_status' in data:
                if 'extended_tweet' in data['retweeted_status']:
                    #if tweet is over the 140 word limit
                    text = data['retweeted_status']['extended_tweet']['full_text']
                else:
                    text = data['retweeted_status']['text']
            else:
                # Else if a normal Tweeet
                if 'extended_tweet' in data:
                    # If tweet is over 140 word limit
                    text = data['extended_tweet']['full_text']
                else:
                    text = data['text']
            
            tweet = utilityFuncs().cleanTweet(text)
            tweetText = utilityFuncs().removeSpacing(tweet[0])

            tweet = tweetText+' '+tweet[1]

            print(tweet)

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
        except BaseException as e:
            print("Error: %s" % str(e))
        return True
          
    def on_error(self, status_code):
        if status_code == 420:
            return False
        print(status_code)

 
if __name__ == '__main__':
 
    hashtag = ["Bitcoin", "bitcoin"]
    tweets_file = "tweets.json"
    tweet_data = []

    twitter_streamer = Streamer()
    twitter_streamer.stream_tweets(tweets_file, hashtag)
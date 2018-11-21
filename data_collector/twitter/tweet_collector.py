import tweepy
import pandas, os

from dotenv import load_dotenv
from pathlib import Path  # python3 only
env_path = Path('.') / 'twitter/config/twitter.env'
load_dotenv(dotenv_path=env_path)

class keys():

    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv("API_SECRET")
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.access_secret = os.getenv("ACCESS_SECRET")

auth = tweepy.OAuthHandler(keys().api_key, keys().api_secret)
auth.set_access_token(keys().access_token, keys().access_secret)

api = tweepy.API(auth)

def collector():
    #Collect tweets from #bitcoin
    print()

if __name__=='__main__':
    print("=== tweet_collector.py ===")
    collector()

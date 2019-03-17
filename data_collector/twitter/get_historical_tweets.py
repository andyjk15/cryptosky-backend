## Might need to train as 2500 tweets will only be collected due to rate limits

import requests as req
import os, json, re, sys, csv
import pandas as pd
import numpy as np

import datetime

import tweet_collector
import spam_filter
import analysis_engine.sentiment_analysis as sentiment_analysis

from dotenv import load_dotenv
from pathlib import Path  # python3 only
env_path = Path('.') / 'data_collector/twitter/config/twitter.env'
load_dotenv(dotenv_path=env_path)

def request(dates):

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    auth = api_key + ':' + api_secret

    bearer_token= req.get('https://api.twitter.com/oauth2/token', data={'grant_type=client_credentials'}, auth={auth})

    bearer_header = 'authorization: Bearer ' + bearer_token

    #split dates for
    for line in dates:
        date = line.split(',')

        response = req.post('https://api.twitter.com/1.1/tweets/search/fullarchive/test.json', \
            headers={bearer_header, 'content-type: application/json'}, \
            data={"query": "bitcoin", "tag": "bitcoin", "fromDate": date[0], "toDate": date[1], "maxResults": 100})

        results = response["results"]

        processTweet(results)

def processTweet(results):
    print("Console: ", "Training Spam Filter...")
    sys.stdout.flush()

    tweetFilter = tweet_collector.filterSpam('data_collector/spam_ham.csv')
    tweetFilter.trainFilter()

    prediction = tweetFilter.testData_Prediction()
    #print("Console: ", "Prediction - ", prediction)

    tweetFilter.filterStatistics(prediction)

    print("Console: ", "Initialising Analysis Engine...")
    sys.stdout.flush()

    analyser = sentiment_analysis.get_sentiment()
    print("Console: ", "Updating lexicon with new words and sentiment...")
    sys.stdout.flush()

    analyser.set_newSentiment()

    for text in results:
        tweet = text['text']

        removedLines = tweet_collector.utilityFuncs().fixLines(text)
        removedSpecialChars = tweet_collector.utilityFuncs().cleanTweet(removedLines)
        removedSpacing = tweet_collector.utilityFuncs().removeSpacing(removedSpecialChars[0])
        tweetLength = tweet_collector.utilityFuncs().checkLength(removedSpacing)

        if tweetLength == True:

            checkIfEnglish = tweet_collector.utilityFuncs().detectLaguage(removedSpecialChars[0])


            if checkIfEnglish == True:

                tweetText = tweet_collector.utilityFuncs().remove_non_ascii(removedSpacing)
                print("Cleaned Tweet: ", tweetText)
                sys.stdout.flush()
 
                cleanedTweet = tweetText+' '+removedSpecialChars[1]

                ## Check with spam filter
                classification = tweetFilter.testTweet(cleanedTweet)

                if classification == False:
                    ## Perform Sentiment Analysis
                    ovSentiment, compound = analyser.get_vader_sentiment(cleanedTweet)

                    try:
                            
                        with open('data_collector/historical_tweets.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                            writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:%M"), 'tweet': cleanedTweet, 'sentiment': ovSentiment, 'compound': compound})

                        return True
                    except BaseException as exception:
                        print("Error: %s" % str(exception))
                        sys.stdout.flush()
                        return False
                else:
                    print("Console: ", "Tweet is spam. Not storing tweet in dataset")
                    sys.stdout.flush()
            else:
                print("Console: ", "Dropping tweet as it is not English")
                sys.stdout.flush()
        else:
            print("Console: ", "Tweet too short for analysis")
            sys.stdout.flush()

def alt_request(alt_dates, new_dates):

    api_key = os.getenv("ALT_API_KEY")
    api_secret = os.getenv("ALT_API_SECRET")

    auth = api_key + ':' + api_secret

    bearer_token= req.get('https://api.twitter.com/oauth2/token', data={'grant_type=client_credentials'}, auth={auth})

    bearer_header = 'authorization: Bearer ' + bearer_token

    for line in alt_dates:
        date = line.split(',')

        response = req.post('https://api.twitter.com/1.1/tweets/search/fullarchive/alt.json', \
            headers={bearer_header, 'content-type: application/json'}, \
            data={"query": "bitcoin", "tag": "bitcoin", "fromDate": date[0], "toDate": date[1], "maxResults": 100})
        
        results = response["results"]

        processTweet(results)

    for i, line in enumerate(new_dates):
        response = req.post('https://api.twitter.com/1.1/tweets/search/fullarchive/alt.json', \
            headers={bearer_header, 'content-type: application/json'}, \
            data={"query": "bitcoin", "tag": "bitcoin", "fromDate": new_dates[i], "toDate": new_dates[i+1], "maxResults": 100})

        results = response["results"]

        processTweet(results)

if __name__ == "__main__":
    # DO 43 reqs then user other API keys
    dates = {
        '201801070000,201801110000',
        '201801120000,201801160000',
        '201801170000,201801210000',
        '201801220000,201801260000',
        '201801270000,201801310000',
        '201802010000,201802050000',
        '201802060000,201802100000',
        '201802110000,201802150000',
        '201802160000,201802200000',
        '201802210000,201802250000',
        '201802260000,201803020000',
        '201803060000,201803100000',
        '201803110000,201803150000',
        '201803160000,201803200000',
        '201803210000,201803250000',
        '201803260000,201803300000',
        '201804010000,201804050000',
        '201804060000,201804100000',
        '201804110000,201804150000',
        '201804160000,201804200000',
        '201804210000,201804250000',
        '201804260000,201804300000',
        '201805010000,201805050000',
        '201805060000,201805100000',
        '201805110000,201805150000',
        '201805160000,201805200000',
        '201805210000,201805250000',
        '201805260000,201805300000',
        '201805310000,201806040000',
        '201806050000,201806090000',
        '201806100000,201806140000',
        '201806150000,201806190000',
        '201806200000,201806240000',
        '201806250000,201806290000',
        '201806300000,201807040000',
        '201807050000,201807090000',
        '201807100000,201807140000',
        '201807150000,201807190000',
        '201807200000,201807240000',
        '201807250000,201807290000',
        '201807300000,201808030000',
        '201808040000,201808080000',
        '201808090000,201808130000'
    }

    alt_dates = {
        '201808140000,201808180000',
        '201808190000,201808230000',
        '201808240000,201808280000',
        '201808290000,201809020000',
        '201809030000,201809070000',
        '201809080000,201809120000',
        '201809130000,201809170000',
        '201809180000,201809220000',
        '201809230000,201809270000',
        '201809280000,201810020000',
        '201810030000,201810070000',
        '201810080000,201810120000',
        '201810130000,201810170000',
        '201810180000,201810220000',
        '201810230000,201810270000',
        '201810280000,201811010000',
        '201811020000,201811060000',
        '201811070000,201811110000',
        '201811120000,201811160000',
        '201811200000,201811240000',
        '201811250000,201811290000',
        '201811300000,201812040000',
        '201812050000,201812090000',
        '201812100000,201812140000',
        '201812150000,201812190000',
        '201812200000,201812240000',
        '201812250000,201812290000',
        '201812300000,201901030000',
        '201901040000,201901080000',
        '201901090000,201901130000',
        '201901140000,201901180000',
        '201901190000,201901230000',
        '201901240000,201901280000',
        '201901290000,201902020000',
        '201902030000,201902070000',
        '201902080000,201902120000',
        '201902130000,201902170000',
        '201902180000,201902220000',
        '201902230000,201902270000'
    }

    #NEED TO FILL IN MISSING PRICE DATA UP TO THE 18/03!!!!!!! - NEED TO KEEP PRICE AND TWEET GATHERING RUNNING (CALL A PRICE EVERTIME THERES A TWEETS SO
    # AMOUNT IS CONSISTANT)

    new_dates = pd.date_range(start='2019-02-28', end='2019-03-19', closed=None)
    dates = {}

    for i, line in enumerate(new_dates):
        line = line.strftime('%Y%m%d')
        dates[i] = line

    print(dates)

    with open('data_collector/historical_tweets.csv', mode='w') as csv_file:
        fieldnames = ['created_at', 'tweet', 'sentiment', 'compound']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

    processTweet("adda ad")
    #print(tweet)
    #request(dates)
    #alt_request(alt_dates, dates)
import os, sys, time, csv
import pandas as pd

from datetime import datetime, timedelta
from time import sleep
import subprocess

if __name__ == "__main__":
    print("Cryptosky Predictions")

    with open('data_collector/live_sentiment.csv', mode='w') as csv_file:
        live_fieldnames = ['created_at', 'compound']
        writer = csv.DictWriter(csv_file, fieldnames=live_fieldnames)

        writer.writeheader()

    ## Run scripts in background detatched
    with open('price_out.log', mode='a') as price:
        price_collector = subprocess.Popen(["python3", "-u","data_collector/prices/price_collector.py"], stdout = price)

    sleep(30)

    with open('lstm_out.log', mode='a') as lstm:
        LSTM_Network = subprocess.Popen(["python3", "-u", "prediction_engine/LSTM.py"], stdout = lstm)
    
    #sleep(3600)
    while True:
        ## Loop tweet collector for an hour

        #Run attached process for tweet collector
        with open('tweet_out.log', mode='a') as tweet:
            state = subprocess.Popen(["python3", "data_collector/twitter/tweet_collector.py"], stdin=subprocess.PIPE, stdout = tweet)
            stdout, stderr = state.communicate()



            print(state.returncode)
            sleep(60)
            if state.returncode == 0:
                print("Hour Passed")
                hour_tweets = pd.read_csv('data_collector/temp_tweets.csv')

                hour_tweets = hour_tweets.drop_duplicates()

                mean_compound = hour_tweets['compound'].mean()
                
                live_time = datetime.now() + timedelta(hours=1)
                print(live_time)
                try:
                    with open('data_collector/live_sentiment.csv', mode='a') as live:
                        writer = csv.DictWriter(live, fieldnames=live_fieldnames)
                        writer.writerow({'created_at': live_time.strftime("%Y-%m-%d %H:00:00"), 'compound': mean_compound})
                except BaseException as exception:
                    print("1 Error: %s" % str(exception))
                    sys.stdout.flush()

                    ## Wipe temp file
                    with open('data_collector/temp_tweets.csv', mode='w') as csv_file:
                        temp_fieldnames = ['created_at', 'tweet', 'sentiment', 'compound']
                        writer = csv.DictWriter(csv_file, fieldnames=temp_fieldnames)

                        writer.writeheader()

                sleep(540)
            else:
                print("Looping")


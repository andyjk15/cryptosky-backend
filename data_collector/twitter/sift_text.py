import json, os, sys, csv

import tweet_collector
import spam_filter
import analysis_engine.sentiment_analysis as sentiment_analysis
import datetime

def processTweet(tweet, tweetFilter):

    now = datetime.datetime.now()

    removedLines = tweet_collector.utilityFuncs().fixLines(tweet)
    removedSpecialChars = tweet_collector.utilityFuncs().cleanTweet(removedLines)
    removedSpacing = tweet_collector.utilityFuncs().removeSpacing(removedSpecialChars[0])
    tweetLength = tweet_collector.utilityFuncs().checkLength(removedSpacing)

    print(removedSpecialChars[0])

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
                            
                    with open('data_collector/historical_tweets-old.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=['created_at', 'tweet', 'sentiment', 'compound'])
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

if __name__ == "__main__":

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
    #print("Console: ", "Updating lexicon with new words and sentiment...")
    #sys.stdout.flush()

    #analyser.set_newSentiment()

    #with open('data_collector/historical_tweets.csv', mode='w') as csv_file:
    #    fieldnames = ['created_at', 'tweet', 'sentiment', 'compound']
    #    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    #    writer.writeheader()

    with open('data_collector/twitter/temp_hist_tweets.json') as file:
        data = json.load(file)

        print(data)
        data = data['results']

        print(len(data))

        for i, k in enumerate(data):
            tweet = k['text']
            print(tweet)
            processTweet(tweet, tweetFilter)
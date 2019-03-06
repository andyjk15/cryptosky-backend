from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import json
import pandas as pd

class get_sentiment():

    def __init__(self):
        pass

    def get_vader_sentiment(self, sentence, analyser):
        score = analyser.polarity_scores(sentence)
        return score

    def get_textblob_sentiment(self, sentence):
        score = TextBlob(sentence)
        score = score.sentiment.polarity
        sentiment = None

        # Basing sentiment level thresholds on Varder thesholds
        if score > 0.05:
            sentiment = 'positive', score
        elif score > -0.05 and score < 0.05:
            sentiment = 'neutral', score
        elif score < -0.05:
            sentiment = 'negative', score
        else:
            print("Error calculating sentiment")

        return score, sentiment


if __name__ == '__main__':
    print("Console: ", "Running, sentiment_analysis.py...")
    
    print("Console:", "Initialising Vader analyser")
    analyser = SentimentIntensityAnalyzer()

    print("Console: ", "Adding marketing words and sentiment to lexicon...")
    new_sentiment = {
        'bull'      : 1,
        'bear'      : -1,
        'bullish'   : 2.5,
        'bearish'   : -2.5,
        'up'        : 0.5,
        'down'      : -0.5,
        'high'      : 2.0,
        'low'       : -2.0,
        'higher'    : 1.8,
        'lower'     : -1.8,
        'absconded' : -2.0,
        'maximalists' : -2.0,
        'regulate' : -0.3,
        'infamous'  : 1.2,
        'trade higher' : 1.0,
        'trade lower'   : -1.0,
        'revival'   : 2.8,
        'centralized' : -1.2,
        'decentralized' : 1.2,
        'centralised' : -1.2,
        'decentralised' : 1.2,
        'decentralization' : 1.3,
        'decentralisation' : 1.3,
        'centralization' : -1.3,
        'centralisation' : -1.3,
        'bans' : -2.6,
        'hodl' : 1.8,
        'ambiguity' : -2.4,
        'revolutionize' : 2.1,
        'revolutionise' : 2.1,
        'consolidation' : 2.5,
        'shorts' : -1.3,
        'longs' : 1.3,
        'long' : 2.2,
        'short' : -2.2,
        'shorting' : -2.1,
        'grow' : 1.2,
        'volatile' : -0.9,
        'rally' : 1.9,
        'rallying' : 1.7,
        'noob' : -0.7,
        'noobs' : -0.9,
        'innovation' : 0.4,
        'bottom' : -0.4,
        'top' : 0.4,
        'topped' : 0.5,
        'bottomed' : -0.5,
        'upwards' : 0.7,
        'downwards' : -0.7,
        'invest' : 1.0,
        'raging' : 2.8,
        'rocketing' : 3.1,
        'swing' : 0.3,
        'swinging' : 0.2,
        'stake' : 0.4,
        'whale' : -1.2,
        'whales' : -1.3,
        'lull' : -1.1,
        'moon' : 1.7,
        'choppy' : -0.2,
        
    }

    analyser.lexicon.update(new_sentiment)

    with open('data_collector/tweets.json') as file:
        tweet_data = json.loads(file.read())

        for line in tweet_data:
            sentence = line['text']
            vader_score = get_sentiment().get_vader_sentiment(sentence, analyser)
            textblob_score = get_sentiment().get_textblob_sentiment(sentence)
            print("Vader Sentiment: \n Tweet: ", sentence, "\n Sentiment: ", vader_score)
            print("TextBlob Sentiment: \n Tweet: ", sentence, "\n Sentiment: ", textblob_score)
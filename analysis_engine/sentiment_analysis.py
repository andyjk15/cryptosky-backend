from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import json
import pandas as pd

class get_sentiment(object):

    def __init__(self):
        self.analyser = SentimentIntensityAnalyzer()
        self.sentiment = {}
        self.compound = {}

    def get_vader_sentiment(self, sentence):
        score = self.analyser.polarity_scores(sentence)

        # Split dict into overall sentiment and compound
        sentiment = list(score.values())
        compound = sentiment[3:]
        compound = compound[0]

        sentiment = sentiment[:3]
        
        # Compare and find overall sentiment
        #print(score)

        score = max(sentiment)
        pos = [i for i, j in enumerate(sentiment) if j == score]

        if pos[0] == 1:
            print("Console: ", "Tweet is overal Neutral - Score: ", score)
            # return neg or pos which ever is higher
            if sentiment[0] > sentiment[2]:
                score = sentiment[0]
            else:
                score = sentiment[2]
            return score, compound
        else:
            return score, compound

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

    def set_newSentiment(self):
        print("Console: ", "Adding marketing words and sentiment to lexicon...")
        new_sentiment = {
            'bull'      : 2,
            'bear'      : -2,
            'bullish'   : 3.5,
            'bearish'   : -3.5,
            'up'        : 1.5,
            'down'      : -1.5,
            'high'      : 2.9,
            'low'       : -2.9,
            'higher'    : 2.8,
            'lower'     : -2.8,
            'absconded' : -2.0,
            'maximalists' : -2.4,
            'regulate' : -2.3,
            'infamous'  : 2.2,
            'trade higher' : 2.0,
            'trade lower'   : -2.0,
            'revival'   : 2.8,
            'centralized' : -2.2,
            'decentralized' : 2.2,
            'centralised' : -2.2,
            'decentralised' : 2.2,
            'decentralization' : 2.3,
            'decentralisation' : 2.3,
            'centralization' : -2.3,
            'centralisation' : -2.3,
            'bans' : -2.6,
            'hodl' : 2.8,
            'ambiguity' : -2.4,
            'revolutionize' : 2.1,
            'revolutionise' : 2.1,
            'consolidation' : 2.5,
            'shorts' : -2.3,
            'longs' : 2.3,
            'long' : 2.2,
            'short' : -2.2,
            'shorting' : -2.8,
            'grow' : 2.2,
            'volatile' : -1.9,
            'rally' : 2.9,
            'rallying' : 2.7,
            'noob' : -1.7,
            'noobs' : -1.9,
            'innovation' : 1.4,
            'bottom' : -1.4,
            'top' : 1.4,
            'topped' : 1.5,
            'bottomed' : -1.5,
            'upwards' : 1.7,
            'downwards' : -1.7,
            'invest' : 2.0,
            'raging' : 3.0,
            'rocketing' : 3.1,
            'swing' : 1.3,
            'swinging' : 1.2,
            'stake' : 1.4,
            'whale' : -2.2,
            'whales' : -2.3,
            'lull' : -2.1,
            'moon' : 2.7,
            'choppy' : -1.2,
            'buy' : 1.9,
            'buying' : 1.7,
            'sell' : -1.7,
            'selling' : -1.9,
            'start selling' : -2.3,
            'stop selling' : 1.4,
            'start buying' : 2.3,
            'stop buying' : -1.4  
        }

        self.analyser.lexicon.update(new_sentiment)


if __name__ == '__main__':
    print("Console: ", "Running, sentiment_analysis.py...")
    
    print("Console:", "Initialising Vader analyser")
    #analyser = SentimentIntensityAnalyzer()

    #with open('data_collector/tweets.json') as file:
        #tweet_data = json.loads(file.read())

        #for line in tweet_data:
            #sentence = line['text']
            #vader_score = get_sentiment().get_vader_sentiment(sentence, analyser)
            #textblob_score = get_sentiment().get_textblob_sentiment(sentence)
            #print("Vader Sentiment: \n Tweet: ", sentence, "\n Sentiment: ", vader_score)
            #print("TextBlob Sentiment: \n Tweet: ", sentence, "\n Sentiment: ", textblob_score)
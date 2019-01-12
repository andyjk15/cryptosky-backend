from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import json

class get_sentiment():

    def __init__(self):
        pass

    def get_vader_sentiment(self, sentence):
        analyser = SentimentIntensityAnalyzer()
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
    print("sentiment_analysis.py")
    
    with open('data_collector/tweets.json') as file:
        tweet_data = json.loads(file.read())

        for line in tweet_data:
            sentence = line['text']
            vader_score = get_sentiment().get_vader_sentiment(sentence)
            textblob_score = get_sentiment().get_textblob_sentiment(sentence)
            print("Vader Sentiment: \n Tweet: ", sentence, "\n Sentiment: ", vader_score)
            print("TextBlob Sentiment: \n Tweet: ", sentence, "\n Sentiment: ", textblob_score)
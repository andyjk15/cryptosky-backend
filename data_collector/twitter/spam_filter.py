#####################
#
#
#
#
#
#
#
#
########################

import os, sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
#from wordcloud import cloud ## To visualise world counts ## Remove later
from math import log, sqrt
import pandas as pd
import numpy as np
from tqdm import tqdm

#def visualiseSpam(data):

#    spamWords = ' '.join(list(data[data['class'] == 1]['tweets']))
#    spam_count = cloud(width = 512,height = 512).generate(spamWords)
#    plt.figure(figsize = (10, 8), facecolor = 'k')
#    plt.imshow(spam_count)
#    plt.axis('off')
#    plt.tight_layout(pad = 0)
#    plt.show()

#def visualiseHam(data):
    
#    hamWords = ' '.join(list(data[data['class'] == 1]['tweets']))
#    ham_count = cloud(width = 512,height = 512).generate(hamWords)
#    plt.figure(figsize = (10, 8), facecolor = 'k')
#    plt.imshow(ham_count)
#    plt.axis('off')
#    plt.tight_layout(pad = 0)
#    plt.show()

## Logic

def processTweet(tweet, gram = 2):
    tweet = tweet.lower() #Lower cases
    
    words = word_tokenize(tweet)    #Tokenise words in text
    words = [w for w in words if len(w) > 2]

    if gram > 2:    ## Increasing grams can increase accuracy
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w

    # Remove stopwords
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]

    stemmer = PorterStemmer()           # Stem words
    words = [stemmer.stem(word) for word in words]   

    return words

class classifier(object):
    def __init__(self, trainData):
        self.tweet = trainData['tweet']
        self.labels = trainData['class']

    def train(self):
        self.TF_and_IDF()
        self.TF_IDF()

    def TF_and_IDF(self):
        noTweets = self.tweet.shape[0]
        self.spam = self.labels.value_counts()[1]
        self.ham = self.labels.value_counts()[0]
        self.total = self.spam + self.ham

        # Initialise spam vars
        self.spamCount = 0
        self.hamCount = 0
        self.tfSpam = dict()
        self.tfHam = dict()
        self.idfSpam = dict()
        self.idfHam = dict()

        ## Logic

        for entry in range(noTweets):
            processed = processTweet(self.tweet[entry])
            count = list() #To keep track of whether the word has ocured in the message or not. IDF count
            for word in processed:
                if self.labels[entry]:
                    self.tfSpam[word] = self.tfSpam.get(word, 0) + 1
                    self.spamCount += 1
                else:
                    self.tfHam[word] = self.tfHam.get(word, 0) + 1
                    self.hamCount += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[entry]:
                    self.idfSpam[word] = self.idfSpam.get(word, 0) + 1
                else:
                    self.idfHam[word] = self.idfHam.get(word, 0) + 1

    def TF_IDF(self):
        self.probSpam = dict()
        self.probHam = dict()
        self.sumSpam = 0
        self.sumHam = 0
        for word in self.tfSpam:
            self.probSpam[word] = (self.tfSpam[word]) * log((self.spam + self.ham) / (self.idfSpam[word] + self.idfHam.get(word, 0)))
            self.sumSpam += self.probSpam[word]
        for word in self.tfSpam:
            self.probSpam[word] = (self.probSpam[word] + 1) / (self.sumSpam + len(list(self.probSpam.keys())))
        for word in self.tfHam:
            self.probHam[word] = (self.tfHam[word]) * log((self.spam + self.ham) / (self.idfSpam.get(word, 0) + self.idfHam[word]))
            self.sumHam += self.probHam[word]
        for word in self.tfHam:
            self.probHam[word] = (self.probHam[word] + 1) / (self.sumHam + len(list(self.probHam.keys())))
            
        self.probSpamTotal, self.probHamTotal = self.spam / self.total, self.ham / self.total

    def classify(self, processed):
        pSpam, pHam = 0, 0
        for word in processed:                
            if word in self.probSpam:
                pSpam += log(self.probSpam[word])
            else:
                pSpam -= log(self.sumSpam + len(list(self.probSpam.keys())))
            if word in self.probHam:
                pHam += log(self.probHam[word])
            else:
                pHam -= log(self.sumHam + len(list(self.probHam.keys()))) 
            pSpam += log(self.probSpamTotal)
            pHam += log(self.probHamTotal)
        return pSpam >= pHam
    
    def predict(self, testData):
        result = dict()
        for (i, tweet) in enumerate(testData):
            processed = processTweet(tweet)
            result[i] = int(self.classify(processed))
        return result

def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


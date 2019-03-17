import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error                  ## Possibly remove if not needed
from keras.models import Sequential
from keras.layers import Dense, LSTM

import queue

from tqdm import tqdm
from keras_tqdm import TQDMCallback

true_q = queue.Queue()
pred_q = queue.Queue()

class Network(object):

    def __init__(self, tweet_file, price_file):
        self.tweet_file = tweet_file
        self.price_file = price_file

    def data(self):
        self.preprocess()

        ## ONLY DO PRICES

        loopback = 2


        train_X, train_Y = self.create_sets(self.price_train, loopback, sent=False) # May remove sent arg
        test_X, test_Y = self.create_sets(self.price_test, loopback, sent=False)

        train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

        self.model_network(train_X, test_X, train_Y, test_Y)

    def preprocess(self):
        sentiment_data = pd.read_csv(self.tweet_file)
        self.price_data = pd.read_csv(self.price_file)

        sentiment_data = sentiment_data['compound'].values.reshape(-1,1)
        self.price_data = self.price_data['price'].values.reshape(-1,1)


        # Unfortunately, I'd advice you to test your solution to such issues as currently no consistency in data types are guaranteed. 
        # Usually - most of the transformers return data in a provided format so as long your base data is in float32 
        # - it will stay float32. But there are some edge cases like to_categorical.
        sentiment_data = sentiment_data.astype('float32')
        self.price_data = self.price_data.astype('float32')

        scale = MinMaxScaler(feature_range=(0,1))
        scaledPrice = scale.fit_transform(self.price_data)

        self.price_train_size = int(len(scaledPrice) * 0.7 )            # use 70% for training
        self.price_test_size = len(scaledPrice) - self.price_train_size
        #self.sent_train_size = int(len(sentiment_data) * 0.7 )
        #self.sent_test_size = len(sentiment_data) - self.sent_train_size

        #print(self.price_train_size, self.price_test_size)
        #print(self.sent_train_size, self.sent_test_size)

        # Get said train data on size

        self.price_train = scaledPrice[0:self.price_train_size:]
        self.price_test = scaledPrice[self.price_test_size:len(scaledPrice):]

        #self.sent_train = sentiment_data[0:self.sent_train_size:]
        #self.sent_test = sentiment_data[self.sent_test_size:len(sentiment_data):]


    def create_sets(self, data, lookback, sent):
        data_X, data_Y = [], []
        with tqdm(total=len(data) - lookback) as pbar:
            for i in range(len(data) - lookback):
                if i >= lookback:
                    pos = data[i-lookback:i+1, 0]
                    pos = pos.tolist()
                    #if(sent==True):
                        #pos.append(sentiment[i].tolist()[0])
                    data_X.append(pos)
                    data_Y.append(data[i + lookback, 0])
                    pbar.update(1)
        return np.array(data_X), np.array(data_Y)

    def model_network(self, train_X, test_X, train_Y, test_Y):
        model = Sequential()
        model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        history = model.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False, callbacks=[TQDMCallback()])

        yhat = model.predict(test_X)

        scale = MinMaxScaler(feature_range=(0,1))
        scaledPrice = scale.fit_transform(self.price_data)

        yhat_inverse_sent = scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse_sent = scale.inverse_transform(test_Y.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse_sent, yhat_inverse_sent))
        print('Test RMSE: %.3f' % rmse_sent)

        plt.plot(yhat, label='predict')
        plt.plot(test_Y, label='true')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    print("Console: ", "Running Prediction Engine...")

    tweet_file = "data_collector/tweets.csv"
    price_file = "data_collector/prices.csv"

    network = Network(tweet_file, price_file)
    network.data()
    #preprocess(tweet_file, price_file)

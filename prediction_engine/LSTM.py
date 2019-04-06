import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error                  ## Possibly remove if not needed
from keras.models import Sequential
from keras.layers import Dense, LSTM

from time import sleep

import queue
true = queue.Queue()
prediction = queue.Queue()

from tqdm import tqdm
from keras_tqdm import TQDMCallback

class Network(object):

    def __init__(self, merged_lstm_data):
        self.lstm_data = merged_lstm_data

    def data(self):
        self.preprocess()

        loopback = 2

        train_X, train_Y = self.create_sets(self.price_train, loopback, self.sentiment_data[0:self.price_train_size])
        test_X, test_Y = self.create_sets(self.price_test, loopback, self.sentiment_data[self.price_train_size:len(self.scaledPrice)])

        train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

        self.model_network(train_X, test_X, train_Y, test_Y)

    def preprocess(self):
        
        self.model_data = self.lstm_data[['price','compound']].groupby(self.lstm_data['created_at']).mean()

        self.sentiment_data = self.model_data['compound'].values.reshape(-1,1)
        self.price_data = self.model_data['price'].values.reshape(-1,1)

        # Unfortunately, I'd advice you to test your solution to such issues as currently no consistency in data types are guaranteed. 
        # Usually - most of the transformers return data in a provided format so as long your base data is in float32 
        # - it will stay float32. But there are some edge cases like to_categorical.
        self.sentiment_data = self.sentiment_data.astype('float32')
        self.price_data = self.price_data.astype('float32')

        self.scale = MinMaxScaler(feature_range=(0,1))
        self.scaledPrice = self.scale.fit_transform(self.price_data)

        self.price_train_size = int(len(self.scaledPrice) * 0.7 )            # use 70% for training
        self.price_test_size = len(self.scaledPrice) - self.price_train_size
        #self.sent_train_size = int(len(sentiment_data) * 0.7 )
        #self.sent_test_size = len(sentiment_data) - self.sent_train_size

        #print(self.price_train_size, self.price_test_size)
        #print(self.sent_train_size, self.sent_test_size)

        # Get said train data on size

        self.price_train = self.scaledPrice[0:self.price_train_size:]
        self.price_test = self.scaledPrice[self.price_train_size:len(self.scaledPrice):]

        #self.split = 

        #self.sent_train = sentiment_data[0:self.sent_train_size:]
        #self.sent_test = sentiment_data[self.sent_test_size:len(sentiment_data):]


    def create_sets(self, data, lookback, sentiment):
        data_X, data_Y = [], []
        #with tqdm(total=len(data) - lookback) as pbar:
        for i in range(len(data) - lookback):
            if i >= lookback:
                #print("i", i)
                pos = data[i-lookback:i+1, 0]

                #print("sent len", len(sentiment))
                #print("data len", len(data))
                pos = pos.tolist()
                #print("1", pos)

                #print("sentiment to list", sentiment[i].tolist()[0])
                pos.append(sentiment[i].tolist()[0])
                #print("2", pos)

                data_X.append(pos)
                data_Y.append(data[i + lookback, 0])
                #pbar.update(1)
        #print("END")
        return np.array(data_X), np.array(data_Y)

    def model_network(self, train_X, test_X, train_Y, test_Y):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(1))
        self.model.compile(loss='mae', optimizer='adam')

        history = self.model.fit(train_X, train_Y, epochs=500, batch_size=100, validation_data=(test_X, test_Y), verbose=0, shuffle=False, callbacks=[TQDMCallback()])

        yhat = self.model.predict(test_X)

        scale = self.scale
        scaledPrice = self.scaledPrice

        yhat_inverse_sent = scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse_sent = scale.inverse_transform(test_Y.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse_sent, yhat_inverse_sent))
        print('Test RMSE: %.3f' % rmse_sent)

        plt.plot(yhat, label='predict')
        plt.plot(test_Y, label='true')
        plt.legend()
        plt.show()

        #btc_1_trace = go.Scatter(x=self.model_data.index.values[3605-1080-1:], y=yhat_inverse_sent.reshape(1080), name= 'predict_lookup')
        #py.iplot([btc_1_trace])

    def future_trading(self, live_price, live_sentiment):
        previous_val = pd.read_csv('data_collector/historical_prices.csv').tail(1)
        threshold = 0.5



        while True:
            price = pd.read_csv(live_price)
            sentiment = pd.read_csv(live_sentiment)

            price_tail = price.tail(5)
            sentiment_tail = sentiment.tail(5)

            ## Example gets last 5 records for some reason

            price = price_tail['price'].values.reshape(-1,1)
            sentiment = sentiment_tail['sentiment'].values.reshape(-1,1)

            price_scale = self.scale.fit_transform(price)

            testX, testY = self.create_sets(price_scale, 2, sentiment)

            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            yhat = self.model.predict(testX)
            yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))

            true.put(price)
            prediction.put(yhat_inverse[0])

            current_val = ((yhat_inverse[0][0]-previous_val)/previous_val)*100

            if current_val > threshold:
                print("Buy")
            elif current_val <- threshold:
                print("Sell")
            else:
                print("Prediction Error!")

            previous_val = yhat_inverse[0][0]

            sleep(3600)


if __name__ == "__main__":
    print("Console: ", "Running Prediction Engine...")

    tweet_file = pd.read_csv("data_collector/historical_tweets.csv")
    price_file = pd.read_csv("data_collector/historical_prices.csv")

    live_price = "data_collector/live_prices.csv"
    live_sentiment = "data_collector/live_sentiment.csv"


    print("price length", len(price_file))
    price_file.columns = ["created_at","price"]

    print("sent length", len(tweet_file))
    tweet_file.columns = ["created_at","tweet","sentiment","compound"]
    
    merged = pd.merge(left=price_file, right=tweet_file, how="inner")
    print("merge length", len(merged))
    merged.to_csv('merged_lstm_data.csv')

    network = Network(merged)
    network.data()

    network.future_trading(live_price, live_sentiment)
    #preprocess(tweet_file, price_file)

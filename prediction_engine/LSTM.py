import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error                  ## Possibly remove if not needed
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from time import sleep
import datetime

import csv

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

        loopback = 2        #MIGHT NEED TO JUSTIFY

        train_X, train_Y = self.create_sets(self.price_train, loopback, self.sentiment_data[0:self.price_train_size])
        test_X, test_Y = self.create_sets(self.price_test, loopback, self.sentiment_data[self.price_train_size:len(self.scaledPrice)])

        #print(self.test_X)

        train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

        self.model_network(train_X, train_Y, test_X, test_Y)

    def preprocess(self):
        
        self.model_data = self.lstm_data[['price','compound']].groupby(self.lstm_data['created_at']).mean()

        #print(self.model_data)

        self.sentiment_data = self.model_data['compound'].values.reshape(-1,1)
        self.price_data = self.model_data['price'].values.reshape(-1,1)

        #print(type(self.sentiment_data))
        # Unfortunately, I'd advice you to test your solution to such issues as currently no consistency in data types are guaranteed. 
        # Usually - most of the transformers return data in a provided format so as long your base data is in float32 
        # - it will stay float32. But there are some edge cases like to_categorical.
        self.sentiment_data = self.sentiment_data.astype('float32')
        self.price_data = self.price_data.astype('float32')

        self.scale = MinMaxScaler(feature_range=(0,1))
        self.scaledPrice = self.scale.fit_transform(self.price_data)

        self.price_train_size = int(len(self.scaledPrice) * 0.7 )            # use 70% for training
        self.price_test_size = len(self.scaledPrice) - self.price_train_size

        # Get said train data on size

        self.price_train = self.scaledPrice[0:self.price_train_size:]
        self.price_test = self.scaledPrice[self.price_train_size:len(self.scaledPrice):]


    def create_sets(self, data, lookback, sentiment):
        data_X, data_Y = [], []
        for i in range(len(data) - lookback):
            if i >= lookback:
                pos = data[i-lookback:i+1, 0]
                pos = pos.tolist()

                pos.append(sentiment[i].tolist()[0])

                data_X.append(pos)
                data_Y.append(data[i + lookback, 0])
        return np.array(data_X), np.array(data_Y)

    def model_network(self, train_X, train_Y, test_X, test_Y):
        self.model = Sequential()

        ## 1st layer - input layer
        self.model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))

        ## 2nd Layer
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))

        ## 3rd Layer
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))

        ## 4th Layer without sequences
        self.model.add(LSTM(100))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        history = self.model.fit(train_X, train_Y, epochs=200, batch_size=1000, validation_data=(test_X, test_Y), verbose=0, shuffle=False, callbacks=[TQDMCallback()])

        yhat = self.model.predict(test_X)

        scale = self.scale
        scaledPrice = self.scaledPrice

        yhat_inverse_sent = scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse_sent = scale.inverse_transform(test_Y.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse_sent, yhat_inverse_sent))
        print('Test RMSE: %.3f' % rmse_sent)
        
        #plt.figure(1)
        plt.plot(test_Y, label='true')
        plt.plot(yhat, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train.png")
        plt.close()

        self.test_Y_updating = test_Y
        self.yhat_updating = yhat
        

    def future_trading(self, live_price, live_sentiment, predictions_file):
        price_file = pd.read_csv('data_collector/historical_prices.csv')
        previous_sent = pd.read_csv('data_collector/historical_tweets.csv')
        
        self.threshold = 0.5
        ## Train for initial 5          ## REALLY HAVE TO REFACT WHEN HAVE TIME

        sleep(3600)

        self.remodel1(price_file, previous_sent, live_price, live_sentiment)

        sleep(3600)

        self.remodel2(price_file, previous_sent, live_price, live_sentiment)

        sleep(3600)

        self.remodel3(price_file, previous_sent, live_price, live_sentiment)

        sleep(3600)

        self.remodel4(price_file, previous_sent, live_price, live_sentiment)
        ## Then switch to looping as 5 exist in data

        sleep(3600)

        while True:
            self.remodel(price_file, previous_sent, live_price, live_sentiment, predictions_file)

            sleep(3600)

    def remodel1(self, price_file, previous_sent, live_price, live_sentiment):

        price = pd.read_csv(live_price)
        sentiment = pd.read_csv(live_sentiment)     

        ## Will only be 1 entry so get 4 from history
        last_4_price = price_file.tail(4)
        last_4_sent = previous_sent.tail(4)

        price_tail = price.tail()
        sentiment_tail = sentiment.tail()

        ## Index fix
        last_4_price.index = last_4_price['created_at']
        price_tail.index = price_tail['created_at']
        last_4_sent.index = last_4_sent['created_at']
        sentiment_tail.index = sentiment_tail['created_at']

        tmp_previous_val = price_file.tail(1)

        self.previous_val = tmp_previous_val['price'].values

        ## Combine price and sents
        price_tail = pd.concat([last_4_price, price_tail], axis=0)
        sentiment_tail = pd.concat([last_4_sent, sentiment_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)
        sentiment = sentiment_tail['compound'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2, sentiment)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)
        
        true.put(price)
        prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
        elif current_val < self.threshold:
            print("Sell")
        else:
            print("Prediction Error!")

        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        #plt.figure(1)
        plt.plot(testY, label='true')
        plt.plot(yhat, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train_1.png")
        plt.close()

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat))

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train.png")
        plt.close()
        
        self.previous_val = yhat_inverse[0][0] ##THE NEXT PREDICTED VALUE IN AN HOUR

        ### Create plot with and without sentiment modeled!!!

    def remodel2(self, price_file, previous_sent, live_price, live_sentiment):

        price = pd.read_csv(live_price)
        sentiment = pd.read_csv(live_sentiment)     

        ## Will only be 1 entry so get 4 from history
        last_3_price = price_file.tail(3)
        last_3_sent = previous_sent.tail(3)

        price_tail = price.tail(2)
        sentiment_tail = sentiment.tail(2)

        last_3_price.index = last_3_price['created_at']
        price_tail.index = price_tail['created_at']
        last_3_sent.index = last_3_sent['created_at']
        sentiment_tail.index = sentiment_tail['created_at']

        ## Combine price and sents
        price_tail = pd.concat([last_3_price, price_tail], axis=0)
        sentiment_tail = pd.concat([last_3_sent, sentiment_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)
        sentiment = sentiment_tail['compound'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2, sentiment)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        true.put(price)
        prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
        elif current_val < self.threshold:
            print("Sell")
        else:
            print("Prediction Error!")
        
        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        self.previous_val = yhat_inverse[0][0]

        #plt.figure(1)
        plt.plot(testY, label='true')
        plt.plot(yhat, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train_2.png")
        plt.close()

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat))

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train.png")
        plt.close()

        #Print current Predictions
        print("Current Predictions: ", list(prediction.queue))

    def remodel3(self, price_file, previous_sent, live_price, live_sentiment):

        price = pd.read_csv(live_price)
        sentiment = pd.read_csv(live_sentiment)     

        ## Will only be 1 entry so get 4 from history
        last_2_price = price_file.tail(2)
        last_2_sent = previous_sent.tail(2)

        price_tail = price.tail(3)
        sentiment_tail = sentiment.tail(3)

        last_2_price.index = last_2_price['created_at']
        price_tail.index = price_tail['created_at']
        last_2_sent.index = last_2_sent['created_at']
        sentiment_tail.index = sentiment_tail['created_at']

        ## Combine price and sents
        price_tail = pd.concat([last_2_price, price_tail], axis=0)
        sentiment_tail = pd.concat([last_2_sent, sentiment_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)
        sentiment = sentiment_tail['compound'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2, sentiment)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        true.put(price)
        prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
        elif current_val < self.threshold:
            print("Sell")
        else:
            print("Prediction Error!")
        
        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        self.previous_val = yhat_inverse[0][0]

        #plt.figure(1)
        plt.plot(testY, label='true')
        plt.plot(yhat, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train_3.png")
        plt.close()

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat))

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train.png")
        plt.close()

        #Print current Predictions
        print("Current Predictions: ", list(prediction.queue))

    def remodel4(self, price_file, previous_sent, live_price, live_sentiment):

        price = pd.read_csv(live_price)
        sentiment = pd.read_csv(live_sentiment)     

        ## Will only be 1 entry so get 4 from history
        last_1_price = price_file.tail()
        last_1_sent = previous_sent.tail()

        price_tail = price.tail(4)
        sentiment_tail = sentiment.tail(4)

        last_1_price.index = last_1_price['created_at']
        price_tail.index = price_tail['created_at']
        last_1_sent.index = last_1_sent['created_at']
        sentiment_tail.index = sentiment_tail['created_at']

        ## Combine price and sents
        price_tail = pd.concat([last_1_price, price_tail], axis=0)
        sentiment_tail = pd.concat([last_1_sent, sentiment_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)
        sentiment = sentiment_tail['compound'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2, sentiment)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        true.put(price)
        prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
        elif current_val < self.threshold:
            print("Sell")
        else:
            print("Prediction Error!")
        
        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        self.previous_val = yhat_inverse[0][0]

        #plt.figure(1)
        plt.plot(testY, label='true')
        plt.plot(yhat, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train_4.png")
        plt.close()

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat))

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train.png")
        plt.close()

        #Print current Predictions
        print("Current Predictions: ", list(prediction.queue))

    def remodel(self, price_file, previous_sent, live_price, live_sentiment, predictions_file):
        price = pd.read_csv(live_price)
        sentiment = pd.read_csv(live_sentiment)

        price_tail = price.tail(5)
        sentiment_tail = sentiment.tail(5)

        price_tail.index = price_tail['created_at']
        sentiment_tail.index = sentiment_tail['created_at']

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)
        sentiment = sentiment_tail['compound'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2, sentiment)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        true.put(price)
        prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100


        if hasattr(Network, 'testY_cont'):
            self.testY_cont = pd.concat([self.testY_cont, testY], axis=0)
            self.yhat_cont = pd.concat([self.yhat_cont, yhat], axis=0)
        else:
            self.testY_cont = testY
            self.yhat_cont = yhat

        if current_val >= self.threshold:
            print("Buy")
        elif current_val < self.threshold:
            print("Sell")
        else:
            print("Prediction Error!")

        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        self.previous_val = yhat_inverse[0][0] ##THE NEXT PREDICTED VALUE IN AN HOUR

        #plt.figure(1)
        plt.plot(self.testY_cont, label='true')
        plt.plot(self.yhat_cont, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train_updating.png")
        plt.close()

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat))

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train.png")
        plt.close()

        #Print current Predictions
        print("Current Predictions: ", list(prediction.queue))

        now = datetime.datetime.now()

        with open(predictions_file, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=predictions_fieldnames)
            writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:00:00"), 'next_hour_price': yhat_inverse[0][0], 'current_price': price_tail.tail(), 'current_sentiment': sentiment_tail.tail()})



if __name__ == "__main__":
    print("Console: ", "Running Prediction Engine...")

    tweet_file = pd.read_csv("data_collector/historical_tweets.csv")
    price_file = pd.read_csv("data_collector/historical_prices.csv")

    live_price = "data_collector/live_prices.csv"
    live_sentiment = "data_collector/live_sentiment.csv"
    predictions_file = "data_collector/predictions.csv"

    price_file.columns = ["created_at","price"]

    tweet_file.columns = ["created_at","tweet","sentiment","compound"]

    ## Create predictions file
    with open(predictions_file, mode='w') as csv_file:
        predictions_fieldnames = ['created_at', 'next_hour_price', 'current_price', 'current_sentiment']
        writer = csv.DictWriter(csv_file, fieldnames=predictions_fieldnames)

        writer.writeheader()
    
    merged = pd.merge(left=price_file, right=tweet_file, how="inner")
    print("merge length", len(merged))
    merged.to_csv('merged_lstm_data.csv')

    network = Network(merged)
    network.data()

    network.future_trading(live_price, live_sentiment, predictions_file)

import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from sklearn.model_selection import StratifiedKFold, cross_val_score

from time import sleep
from datetime import datetime, timedelta

import csv, sys, json

from tqdm import tqdm
from keras_tqdm import TQDMCallback

class Network(object):

    def __init__(self, merged_lstm_data):
        self.lstm_data = merged_lstm_data

    def data(self):
        self.preprocess()

        loopback = 2        #MIGHT NEED TO JUSTIFY

        train_X, train_Y = self.create_sets(self.price_train, loopback)
        test_X, test_Y = self.create_sets(self.price_test, loopback)

        train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

        self.model_network(train_X, train_Y, test_X, test_Y)


    def preprocess(self):

        self.model_data = self.lstm_data[['price']].groupby(self.lstm_data['created_at']).mean()

        self.price_data = self.model_data['price'].values.reshape(-1,1)

        # Unfortunately, I'd advice you to test your solution to such issues as currently no consistency in data types are guaranteed.
        # Usually - most of the transformers return data in a provided format so as long your base data is in float32
        # - it will stay float32. But there are some edge cases like to_categorical.
        self.price_data = self.price_data.astype('float32')

        self.scale = MinMaxScaler(feature_range=(0,1))
        self.scaledPrice = self.scale.fit_transform(self.price_data)

        self.price_train_size = int(len(self.scaledPrice) * 0.7 )            # use 70% for training
        self.price_test_size = len(self.scaledPrice) - self.price_train_size

        # Get said train data on size

        self.price_train = self.scaledPrice[0:self.price_train_size:]
        self.price_test = self.scaledPrice[self.price_train_size:len(self.scaledPrice):]


    def create_sets(self, data, lookback):
        data_X, data_Y = [], []
        for i in range(len(data) - lookback):
            if i >= lookback:
                pos = data[i-lookback:i+1, 0]
                pos = pos.tolist()
                data_X.append(pos)
                data_Y.append(data[i + lookback, 0])
        return np.array(data_X), np.array(data_Y)

    def model_network(self, train_X, train_Y, test_X, test_Y):
        train = 0
        test = 0
        kfold = 0

        testY_inverse_sent, yhat_inverse_sent = self.defineNetwork(train_X, train_Y, test_X, test_Y, train, test, kfold, fold = False)

        #plt.figure(1)
        plt.plot(testY_inverse_sent, label='true')
        plt.plot(yhat_inverse_sent, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_Train.png")
        plt.close()

        self.test_Y_updating = testY_inverse_sent
        self.yhat_updating = yhat_inverse_sent

        cat = np.concatenate((self.test_Y_updating, self.yhat_updating), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_updating.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        ## K-fold
        # kdf = pd.read_csv('merged_lstm_data.csv')
        # kdf = kdf.values()
        # kdf = kdf[['price','compound']].groupby(self.lstm_data['created_at']).mean()

        # X = kdf[:,0:8]
        # Y = kdf[:,8]

        # seed = 7
        # np.random.seed(seed)
        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        # trY, trX = [], []
        # teY, teX = [], []

        
        # #for train, test in kfold.split(train_X, train_Y):
        # testY_inverse_sent, yhat_inverse_sent = self.defineNetwork(X, trY, teX, Y, train, test, kfold, fold = True)
        # print("%.2f%% (+/- %.2f%%) %.2f%%" % (np.mean(self.scoresAr), np.std(self.scoresAr), np.var(self.scoresAr)))
        # kMean = np.mean(self.scoresAr)*100
        # kStd = np.std(self.scoresAr)*100
        # kVar = np.var(self.scoresAr)*100
        # kAcc = np.mean(self.resultsAr)*100
        # kAccStd = np.std(self.resultsAr)*100

        # try:
        #     with open('data/kfold.csv', mode='w') as csv_file:
        #         writer = csv.DictWriter(csv_file, fieldnames=['kMean', 'kStd', 'kVar', 'kAcc', 'kAccStd'])
        #         writer.writerow({'kMean': kMean, 'kStd': kStd, 'kVar': kVar, 'kAcc': kAcc, 'kAccStd': kAccStd})
        # except Exception as e:
        #     print("Error: %s" % str(e))
        #     sys.stdout.flush()

        # exit()
        

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def defineNetwork(self, train_X, train_Y, test_X, test_Y, train, test, kfold, fold):

        if fold == True:
            
            scores = self.model.evaluate(test_X, test_Y, verbose=0)

            yhat = self.model.predict(test_X)

            self.resultsAr = cross_val_score(self.model, train_X, test_Y, cv=kfold, scoring='accuracy')

            self.scoresAr.append(scores[1]*100)

            print("%s: %.2f%%" % (self.model.metrics_names[1], self.scoresAr[1]*100))

            ## Save results for graph

            ar = { i : self.scoresAr[i] for i in range(0, len(self.scoresAr))}
            xs = {}
            with open('data/no_sent_kfold.json', mode='w') as json_file:
                for x in ar:
                    print("BOOP")
                    xs[x] = {'index' : x, 'Acc': ar[x]}
                    print("After")
                json.dump(xs, json_file, indent=3)

            testY_inverse_sent = 0
            yhat_inverse_sent = 0
        else:
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
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape', self.f1_m, self.precision_m, self.recall_m])

            history = self.model.fit(train_X, train_Y, epochs=200, batch_size=1000, validation_data=(test_X, test_Y), verbose=0, shuffle=False, callbacks=[TQDMCallback()])

            yhat = self.model.predict(test_X)

            scale = self.scale
            scaledPrice = self.scaledPrice

            yhat_inverse_sent = scale.inverse_transform(yhat.reshape(-1, 1))
            testY_inverse_sent = scale.inverse_transform(test_Y.reshape(-1, 1))

            rmse_sent = sqrt(mean_squared_error(testY_inverse_sent, yhat_inverse_sent))
            print('Test RMSE: %.3f' % rmse_sent)

            mse_com = history.history['mean_squared_error']
            mae_com = history.history['mean_absolute_error']
            mape_com = history.history['mean_absolute_percentage_error']
            loss_com = history.history['loss']

            f1_com = history.history['f1_m']
            precision_com = history.history['precision_m']
            recall_com = history.history['recall_m']

            mse = np.mean(mse_com)*100
            mae = np.mean(mae_com)*100
            mape = np.mean(mape_com)*10
            loss = np.mean(loss_com)*100
            f1 = np.mean(f1_com)*10
            precision = np.mean(precision_com)*10
            recall = np.mean(recall_com)*10

            try:
                with open('data/no_sent_metrics.csv', mode='w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=['RSME', 'MSE', 'MAE', 'MAPE', 'Loss', 'f1', 'precision', 'recall'])
                    writer.writerow({'RSME': rmse_sent, 'MSE': mse, 'MAE': mae, 'MAPE': mape, 'Loss': loss, 'f1': f1, 'precision': precision, 'recall': recall})
            except Exception as e:
                print("Error: %s" % str(e))
                sys.stdout.flush()

            ## Combine
            xs = {}
            mse_com = { i : mse_com[i] for i in range(0, len(mse_com))}
            mae_com = { i : mae_com[i] for i in range(0, len(mae_com))}
            mape_com = { i : mape_com[i] for i in range(0, len(mape_com))}
            loss_com = { i : loss_com[i] for i in range(0, len(loss_com))}
            f1_com = { i : f1_com[i] for i in range(0, len(f1_com))}
            precision_com = { i : precision_com[i] for i in range(0, len(precision_com))}
            recall_com = { i : recall_com[i] for i in range(0, len(recall_com))}

            with open('data/no_sent_metrics_combined.json', mode='w') as json_file:
                for x in f1_com:
                    xs[x] = {'index' : x, 'mean_squared_error': mse_com[x], 'mean_absolute_error' : mae_com[x], \
                        'mean_absolute_percentage_error': mape_com[x], 'loss': loss_com[x], 'f1': f1_com[x], 'precision': precision_com[x], \
                        'recall': recall_com[x]}
                json.dump(xs, json_file, indent=3)

        return testY_inverse_sent, yhat_inverse_sent

    def future_trading(self, live_price, predictions_file):
        price_file = pd.read_csv('data_collector/historical_prices.csv')
        #previous_sent = pd.read_csv('data_collector/historical_tweets.csv')

        #print("PRICE FILE", price_file)
        
        self.threshold = 0.05
        ## Train for initial 5          ## REALLY HAVE TO REFACT WHEN HAVE TIME

        sleep(3600)

        self.remodel1(price_file, live_price, predictions_file)

        sleep(3600)

        self.remodel2(price_file, live_price, predictions_file)

        sleep(3600)

        self.remodel3(price_file, live_price, predictions_file)

        sleep(3600)

        self.remodel4(price_file, live_price, predictions_file)
        ## Then switch to looping as 5 exist in data

        sleep(3600)

        while True:
            self.remodel(price_file, live_price, predictions_file)

            sleep(3600)

    def remodel1(self, price_file, live_price, predictions_file):

        price = pd.read_csv(live_price)   

        ## Will only be 1 entry so get 4 from history
        last_4_price = price_file.tail(4)

        price_tail = price.tail()

        ## Index fix
        last_4_price.index = last_4_price['created_at']
        price_tail.index = price_tail['created_at']

        tmp_previous_val = price_file.tail(1)

        self.previous_val = tmp_previous_val['price'].values

        price_tail = pd.concat([last_4_price, price_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)
        
        #true.put(price)
        #prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
            self.state = 'BUY'
        elif current_val < self.threshold:
            print("Sell")
            self.state = 'SELL'
        else:
            print("Prediction Error!")

        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        #plt.figure(1)
        plt.plot(testY_inverse, label='true')
        plt.plot(yhat_inverse, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Updating.png")
        plt.close()

        print("1: ", testY_inverse)
        print("1: ", yhat_inverse)
        self.testY_cont = testY_inverse
        self.yhat_cont = yhat_inverse

        print("1 cont: ", self.testY_cont)
        print("1 cont: ", self.yhat_cont)

        cat = np.concatenate((self.testY_cont, self.yhat_cont), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_from_start.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)


        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY_inverse))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat_inverse))

        print("Y updating", self.test_Y_updating, testY_inverse)
        print("hat updating", self.yhat_updating, yhat_inverse)

        cat = np.concatenate((self.test_Y_updating, self.yhat_updating), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_updating.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Train.png")
        plt.close()

        now = datetime.now()# + timedelta(hours=1)
        hour = yhat_inverse[0][0]
        current = price_tail.tail(1)

        print("hour ", hour)
        current = current['price'].item()

        try:
            with open(predictions_file, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['created_at', 'next_hour_price', 'current_price', 'state'])
                writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:00:00"), 'next_hour_price': hour, 'current_price': current, 'state': self.state})
        except Exception as e:
            print("Error: %s" % str(e))
            sys.stdout.flush()

        self.previous_val = yhat_inverse[0][0] ##THE NEXT PREDICTED VALUE IN AN HOUR

    def remodel2(self, price_file, live_price, predictions_file):

        price = pd.read_csv(live_price) 

        ## Will only be 1 entry so get 4 from history
        last_3_price = price_file.tail(3)

        price_tail = price.tail(2)

        last_3_price.index = last_3_price['created_at']
        price_tail.index = price_tail['created_at']

        ## Combine price and sents
        price_tail = pd.concat([last_3_price, price_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        #true.put(price)
        #prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
            self.state = 'BUY'
        elif current_val < self.threshold:
            print("Sell")
            self.state = 'SELL'
        else:
            print("Prediction Error!")
        
        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        self.testY_cont = np.concatenate((self.testY_cont, testY_inverse))  ## remove axis
        self.yhat_cont = np.concatenate((self.yhat_cont, yhat_inverse))
        
        print("2: ", testY_inverse)
        print("2: ", yhat_inverse)
        print("2 cont: ", self.testY_cont)
        print("2 cont: ", self.yhat_cont)
        
        plt.plot(self.testY_cont, label='true')
        plt.plot(self.yhat_cont, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Updating.png")
        plt.close()

        cat = np.concatenate((self.testY_cont, self.yhat_cont), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_from_start.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY_inverse))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat_inverse))

        print("Y updating", self.test_Y_updating, testY_inverse)
        print("hat updating", self.yhat_updating, yhat_inverse)

        cat = np.concatenate((self.test_Y_updating, self.yhat_updating), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_updating.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Train.png")
        plt.close()

        #Print current Predictions
        now = datetime.now()# + timedelta(hours=1)
        hour = yhat_inverse[0][0]
        current = price_tail.tail(1)

        print("hour ", hour)
        current = current['price'].item()

        try:
            with open(predictions_file, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['created_at', 'next_hour_price', 'current_price', 'state'])
                writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:00:00"), 'next_hour_price': hour, 'current_price': current, 'state': self.state})
        except Exception as e:
            print("Error: %s" % str(e))
            sys.stdout.flush()

        self.previous_val = yhat_inverse[0][0]

    def remodel3(self, price_file, live_price, predictions_file):

        price = pd.read_csv(live_price)    

        ## Will only be 1 entry so get 4 from history
        last_2_price = price_file.tail(2)

        price_tail = price.tail(3)

        last_2_price.index = last_2_price['created_at']
        price_tail.index = price_tail['created_at']

        ## Combine price and sents
        price_tail = pd.concat([last_2_price, price_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
            self.state = 'BUY'
        elif current_val < self.threshold:
            print("Sell")
            self.state = 'SELL'
        else:
            print("Prediction Error!")
        
        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        self.testY_cont = np.concatenate((self.testY_cont, testY_inverse))
        self.yhat_cont = np.concatenate((self.yhat_cont, yhat_inverse))

        print("3: ", testY_inverse)
        print("3: ", yhat_inverse)
        print("3 cont: ", self.testY_cont)
        print("3 cont: ", self.yhat_cont)

        #plt.figure(1)
        plt.plot(self.testY_cont, label='true')
        plt.plot(self.yhat_cont, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Updating.png")
        plt.close()

        cat = np.concatenate((self.testY_cont, self.yhat_cont), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_from_start.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY_inverse))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat_inverse))
        
        print("Y updating", self.test_Y_updating, testY_inverse)
        print("hat updating", self.yhat_updating, yhat_inverse)

        cat = np.concatenate((self.test_Y_updating, self.yhat_updating), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_updating.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Train.png")
        plt.close()

        #Print current Predictions
        now = datetime.now()# + timedelta(hours=1)
        hour = yhat_inverse[0][0]
        current = price_tail.tail(1)

        print("hour ", hour)
        current = current['price'].item()

        try:
            with open(predictions_file, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['created_at', 'next_hour_price', 'current_price', 'state'])
                writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:00:00"), 'next_hour_price': hour, 'current_price': current, 'state': self.state})
        except Exception as e:
            print("Error: %s" % str(e))
            sys.stdout.flush()

        self.previous_val = yhat_inverse[0][0]

    def remodel4(self, price_file, live_price, predictions_file):

        price = pd.read_csv(live_price)   

        ## Will only be 1 entry so get 4 from history
        last_1_price = price_file.tail()

        price_tail = price.tail(4)

        last_1_price.index = last_1_price['created_at']
        price_tail.index = price_tail['created_at']

        ## Combine price and sents
        price_tail = pd.concat([last_1_price, price_tail], axis=0)

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        #true.put(price)
        #prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        if current_val >= self.threshold:
            print("Buy")
            self.state = 'BUY'
        elif current_val < self.threshold:
            print("Sell")
            self.state = 'SELL'
        else:
            print("Prediction Error!")
        
        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        print("4: ", testY_inverse)
        print("4: ", yhat_inverse)
        print("4 cont: ", self.testY_cont)
        print("4 cont: ", self.yhat_cont)

        self.testY_cont = np.concatenate((self.testY_cont, testY_inverse))
        self.yhat_cont = np.concatenate((self.yhat_cont, yhat_inverse))

        #plt.figure(1)
        plt.plot(self.testY_cont, label='true')
        plt.plot(self.yhat_cont, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Updating.png")
        plt.close()

        cat = np.concatenate((self.testY_cont, self.yhat_cont), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_from_start.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY_inverse))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat_inverse))

        print("Y updating", self.test_Y_updating, testY_inverse)
        print("hat updating", self.yhat_updating, yhat_inverse)

        cat = np.concatenate((self.test_Y_updating, self.yhat_updating), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_updating.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Train.png")
        plt.close()

        #Print current Predictions
        now = datetime.now()# + timedelta(hours=1)
        hour = yhat_inverse[0][0]
        current = price_tail.tail(1)

        print("hour ", hour)
        current = current['price'].item()

        try:
            with open(predictions_file, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['created_at', 'next_hour_price', 'current_price', 'state'])
                writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:00:00"), 'next_hour_price': hour, 'current_price': current, 'state': self.state})
        except Exception as e:
            print("Error: %s" % str(e))
            sys.stdout.flush()

        self.previous_val = yhat_inverse[0][0]

    def remodel(self, price_file, live_price, predictions_file):
        price = pd.read_csv(live_price)

        price_tail = price.tail(5)

        price_tail.index = price_tail['created_at']

        ## Example gets last 5 records for some reason

        price = price_tail['price'].values.reshape(-1,1)

        price_scale = self.scale.fit_transform(price)

        testX, testY = self.create_sets(price_scale, 2)

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        yhat = self.model.predict(testX)
        yhat_inverse = self.scale.inverse_transform(yhat.reshape(-1, 1))
        testY_inverse = self.scale.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
        print('Test RMSE: %.3f' % rmse_sent)

        #true.put(price)
        #prediction.put(yhat_inverse[0])

        current_val = ((yhat_inverse[0][0]-self.previous_val)/self.previous_val)*100

        print("Current Value : ", current_val)

        self.testY_cont = np.concatenate((self.testY_cont, testY_inverse))
        self.yhat_cont = np.concatenate((self.yhat_cont, yhat_inverse))

        if current_val >= self.threshold:
            print("Buy")
            self.state = 'BUY'
        elif current_val < self.threshold:
            print("Sell")
            self.state = 'SELL'
        else:
            print("Prediction Error!")

        print("Predicted Price for next hour: ", yhat_inverse[0][0])

        #plt.figure(1)
        plt.plot(self.testY_cont, label='true')
        plt.plot(self.yhat_cont, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_updating.png")
        plt.close()

        print("Y cont", self.testY_cont)
        print("hat cont", self.yhat_cont)

        cat = np.concatenate((self.testY_cont, self.testY_cont), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_from_start.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        ## Updating plot
        self.test_Y_updating = np.concatenate((self.test_Y_updating, testY_inverse))
        self.yhat_updating = np.concatenate((self.yhat_updating, yhat_inverse))

        print("Y updating", self.test_Y_updating, testY_inverse)
        print("hat updating", self.yhat_updating, yhat_inverse)

        cat = np.concatenate((self.test_Y_updating, self.yhat_updating), axis=1)
        cat = cat.tolist()
        xs = {}
        with open('data/no_sent_updating.json', mode='w') as file:
            for x in range(len(cat)):
                xs[x] = {'index' : x, 'testY_inverse': cat[x][0], 'yhat_inverse' : cat[x][1]}
            json.dump(xs, file, indent=3)

        #plt.figure(2)
        plt.plot(self.test_Y_updating, label='true')
        plt.plot(self.yhat_updating, label='predict')
        plt.title("Bitcoin Price Predictions")
        plt.xlabel("Time - Hours")
        plt.ylabel("Price")
        plt.grid(axis='y', linestyle='-')
        plt.legend()
        plt.savefig("True_Pred_no_sent_Train.png")
        plt.close()

        ## Output plots to jsons

        now = datetime.now()# + timedelta(hours=1)
        hour = yhat_inverse[0][0]
        current = price_tail.tail()

        print("current ", current)
        current = current['price'].item()

        try:
            with open(predictions_file, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['created_at', 'next_hour_price', 'current_price', 'state'])
                writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:00:00"), 'next_hour_price': hour, 'current_price': current, 'state': self.state})
        except Exception as e:
            print("Error: %s" % str(e))
            sys.stdout.flush()

        self.previous_val = yhat_inverse[0][0] ##THE NEXT PREDICTED VALUE IN AN HOUR

if __name__ == "__main__":
    print("Console: ", "Running Prediction Engine...")

    price_file = pd.read_csv("data_collector/historical_prices.csv")

    live_price = "data_collector/live_prices.csv"
    predictions_file = "data/predictions_notsent.csv"

    price_file.columns = ["created_at","price"]

    network = Network(price_file)
    network.data()

    network.future_trading(live_price, predictions_file)

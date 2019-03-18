import pandas as pd
import numpy as np
import csv, datetime
import datetime


if __name__ == "__main__":
    price_file = 'data_collector/historical_prices.csv'

    with open(price_file, mode='w') as csv_file:
        fieldnames = ['created_at', 'price']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


    print("Reading BIGGUN file")

    data = pd.read_csv('BigDATA/coinbaseUSD.csv', skiprows=32000020)

    data.columns=["timestamp", "price", "dunno"]

#    print(data)
    data = data.set_index(['timestamp'])
    data.index = pd.to_datetime(data.index, unit='s')
    
    data.drop("dunno", axis=1, inplace=True)

#    print(data)

    data = data.reset_index().set_index('timestamp').resample('1H').mean()

#    print(data)

    #print(data.isnull().values.any())

    data.price = data.price.fillna((data.price.shift(23) + data.price.shift(-24))/2)

    #print("After NANS", data)
    #print(data[pd.isnull(data).any(axis=1)])

    ## Export to CSV
    data.to_csv(price_file, sep=',')
    
    ##
    new_prices = pd.read_csv('BigDATA/new_daily_BTC_prices.csv')


    new_prices.drop(columns=["Currency", "24h Open (USD)", "24h High (USD)", "24h Low (USD)"], axis=1, inplace=True)
    new_prices.columns = ["timestamp", "price"]

    new_prices['timestamp'] = pd.to_datetime(new_prices['timestamp'])

    data = new_prices.set_index('timestamp').resample('1D').mean().resample('1H').mean()
    data = data.fillna(method='backfill')
    
    print(data)

    with open(price_file, 'a') as file:
        data.to_csv(file, sep=',')

    #print(new_prices.loc[56]['price'])
    
    #data = pd.date_range(start='2019-01-08', end='2019-03-19', closed=None, freq='1H')
    #data = data.values.T.tolist()

    #print(data)

    #prices = pd.DataFrame([], columns=["date", "price"])
    #temp = []
    #stack = []

    #print("PRICES: ", new_prices)
    
    #k=0
    #for i, line in enumerate(data):
    #    print(i)
    #    print(k) 
    #    line = line.strftime('%Y-%m-%d %H-%M-%S')
        #if [d.date() for d in new_prices.loc[k]['timestamp']] == '00-00-00' or [d.date() for d in new_prices.loc[k]['timestamp']] == '23:59:59':
    #    if  new_prices.loc[k]['timestamp'].split()[1] == '23-00-00' or new_prices.loc[k]['timestamp'].split()[1] == '22:59:59':
    #        price = new_prices.loc[k]['price']
    #        temp.append(tuple((line, price)))
    #        stack.append(price)
    #        k += 1

    #        print("ADDING", stack)
    #    else:
    #        print("CHECK", stack)
    #        if len(stack) <= 1:
    #            price = stack[0]
    #            temp.append(tuple((line, price)))
    #        else:
    #            price = (stack[0] + stack[1]) / 2
    #            temp.append(tuple((line, price)))
    #        stack.pop(0)    

    #print("TEMP", temp)
    #print(prices)
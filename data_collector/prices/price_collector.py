#!/usr/bin/env python

#Historical price collector
import requests, json, os
import datetime
from time import sleep

from coinbase.wallet.client import Client


from dotenv import load_dotenv
from pathlib import Path  # python3 only
env_path = Path('.') / 'prices/config/coinbase.env'
load_dotenv(dotenv_path=env_path)

class keys():

    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv("API_SECRET")


def coinbase():

    api_key = keys().api_key
    api_secret = keys().api_secret

    client = Client(api_key, api_secret)
    repsonse = client.get_spot_price(currency_pair = 'BTC-USD')

    price = (float(repsonse['amount']))
    price = round(price, 3)

    return price

def bitfinex():
    response = requests.request("GET", "https://api.bitfinex.com/v1/pubticker/btcusd")
    response = json.loads(response.text)

    price = (float(response['low'])+ float(response['mid']) + float(response['high']))/3
    price = round(price, 3)
    return price

def gemini():
    response = requests.request("GET", "https://api.gemini.com/v1/pubticker/btcusd")
    response = json.loads(response.text)
    
    price = (float(response['last']) + float(response['ask']) + float(response['bid']))/3
    price = round(price, 3)
    return price

def collector():

    averagePrice = (coinbase() + bitfinex() + gemini())/3
    averagePrice = round(averagePrice, 3)

    now = datetime.datetime.now()

    print("Loading existing data")
    with open('../data.json') as file:
        data = json.load(file)
        data.append({
        'Date' : now.strftime("%Y-%m-%d %H:%M:%S"),
        'Symbol' : 'BTC-USD',
        'Price' : averagePrice
    })
    print("Saving data to file")
    with open('../data.json', 'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)


if __name__=='__main__':
    print('== Historical Price Collector ==')
    while True:
        sleep(3600)
        collector()
        print("Complete!")
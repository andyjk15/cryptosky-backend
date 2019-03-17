#!/usr/bin/env python

#Historical price collector
import requests, os, sys, csv, json
import datetime

from time import sleep

from coinbase.wallet.client import Client

from dotenv import load_dotenv
from pathlib import Path  # python3 only
env_path = Path('.') / 'data_collector/prices/config/coinbase.env'
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

    #print(response)

    price = (float(response['low'])+ float(response['mid']) + float(response['high']))/3
    price = round(price, 3)
    return price

def gemini():
    response = requests.request("GET", "https://api.gemini.com/v1/pubticker/btcusd")
    response = json.loads(response.text)

    price = (float(response['last']) + float(response['ask']) + float(response['bid']))/3
    price = round(price, 3)
    return price

def collector(priceCSV):

    now = datetime.datetime.now()

    averagePrice = (coinbase() + bitfinex() + gemini())/3
    averagePrice = round(averagePrice, 3)

    try:
        with open(priceCSV, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:%M"), 'price': averagePrice})

        return True
    except BaseException as exception:
        print("Error: %s" % str(exception))
        sys.stdout.flush()
        return False


if __name__=='__main__':
    print("Console: ", "== Historical Price Collector ==")

    priceCSV = 'data_collector/prices.csv'

    print("Console: ", "Initialising Prices CSV...")
    sys.stdout.flush()

    with open(priceCSV, mode='w') as csv_file:
        fieldnames = ['created_at', 'price']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    while True:
        sleep(300)
        collector(priceCSV)
        #print("Complete!")

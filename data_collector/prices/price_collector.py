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

    try:
        client = Client(api_key, api_secret)
        repsonse = client.get_spot_price(currency_pair = 'BTC-USD')
        price = (float(repsonse['amount']))
        price = round(price, 3)
        return price
    except KeyError as e:
        print("Error: %s" % str(e))
        sys.stdout.flush()
        price = 0
        return price

def bitfinex():

    try:
        response = requests.request("GET", "https://api.bitfinex.com/v1/pubticker/btcusd")
        response = json.loads(response.text)

        price = (float(response['low'])+ float(response['mid']) + float(response['high']))/3
        price = round(price, 3)
        return price
    except KeyError as e:
        print("Error: %s" % str(e))
        sys.stdout.flush()
        price = 0
        return price
    
def gemini():

    try:
        response = requests.request("GET", "https://api.gemini.com/v1/pubticker/btcusd")
        response = json.loads(response.text)

        price = (float(response['last']) + float(response['ask']) + float(response['bid']))/3
        price = round(price, 3)
        return price
    except KeyError as e:
        print("Error: %s" % str(e))
        sys.stdout.flush()
        price = 0
        return price

def collector(priceCSV, fieldnames):

    now = datetime.datetime.now()

    coinbase_P = coinbase()
    bitfinex_P = bitfinex()
    gemini_P = gemini()

    if coinbase_P == 0 or bitfinex_P == 0 or gemini_P == 0:
        if coinbase_P and bitfinex_P == 0:
            averagePrice = gemini_P
            return
        elif coinbase_P and gemini_P == 0:
            averagePrice = bitfinex_P
            return
        elif bitfinex_P and gemini_P == 0:
            averagePrice = coinbase_P
            return
        averagePrice = (coinbase_P + bitfinex_P + gemini_P)/2
    else:
        averagePrice = (coinbase_P + bitfinex_P + gemini_P)/3

    averagePrice = round(averagePrice, 3)

    print("Price: ", averagePrice)

    try:
        with open(priceCSV, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'created_at': now.strftime("%Y-%m-%d %H:00:00"), 'price': averagePrice})

        return True
    except BaseException as exception:
        print("Error: %s" % str(exception))
        sys.stdout.flush()
        return False


if __name__=='__main__':
    print("Console: ", "== Historical Price Collector ==")

    priceCSV = 'data_collector/live_prices.csv'

    print("Console: ", "Initialising Prices CSV...")
    sys.stdout.flush()

    with open(priceCSV, mode='w') as csv_file:
        fieldnames = ['created_at', 'price']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    while True:
        sleep(3600)
        collector(priceCSV, fieldnames)
        #print("Complete!")

import csv, sys
import pandas as pd

if __name__ == "__main__":
    
    data = pd.read_csv('data_collector/historical_tweets.csv')

    dates = pd.date_range(start='2018-01-06 19:00:00', end='2019-04-04 17:00:00', freq='H', closed=None)

    print(len(data))
    print(len(dates))

    boop = {}

    for i, line in enumerate(dates):
        line = line.strftime('%Y-%m-%d %H:%M:%S')
        boop[i] = line

    temp = pd.DataFrame.from_dict(boop, orient='index')
    temp.columns = ['time']

    print(temp['time'])

    empty = pd.DataFrame(temp, columns= ['created_at','tweet','sentiment','compound'])
    
    empty['created_at'] = temp['time']
    empty['tweet'] = data['tweet']
    empty['sentiment'] = data['sentiment']
    empty['compound'] = data['compound']

    print(empty)

    file = 'data_collector/hist_tweets_temp.csv'

    #with open(file, mode='w') as csv_file:
    #    fieldnames = ['created_at', 'tweet', 'sentiment', 'compound']
    #    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #    writer.writeheader()

    with open(file, 'a') as file:
        empty.to_csv(file, sep=',', index=False)
    #alt = data.index.resize(len(dates), 4)

    #alt = pd.DataFrame(data.index.reshape(937, 4))
    #print(alt)
    #data['created_at'] = dates

    #print(data)

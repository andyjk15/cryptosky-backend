import pandas as pd
import numpy as np


if __name__ == "__main__":
    print("Reading BIGGUN file")

    data = pd.read_csv('BigDATA/coinbaseUSD.csv', skiprows=32000020)

    data.columns=["timestamp", "price", "dunno"]

    print(data)
    data = data.set_index(['timestamp'])
    data.index = pd.to_datetime(data.index, unit='s')
    
    data.drop("dunno", axis=1, inplace=True)

    print(data)

    data = data.reset_index().set_index('timestamp').resample('1H').mean()

    print(data)

    print(data.isnull().values.any())

    data.price = data.price.fillna((data.price.shift(23) + data.price.shift(-24))/2)

    print("After NANS", data)
    print(data[pd.isnull(data).any(axis=1)])

    ## Export to CSV


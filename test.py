from aisquared.model import AISquared
import pandas as pd
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def load_dataset():
    # Configuring params
    SEQ_LEN = 60 # Use the last 60 mins to predict
    FUTURE_PERIOD_PREDICT = 1 # Predict the price for the next min

    def normalize(data, mi, ma):
        return (data-mi)/(ma-mi)
    def preprocess_df(df, coefs=None, shuffle=True):
        # Check if coeficients is not empty
        if coefs == None:
            coefs = {}
            for col in df.columns:  # go through all of the columns
                mi, ma  = df[col].min(), df[col].max()
                coefs[col] = {
                    "min": mi,
                    "max": ma
                }
                df[col] = normalize(df[col].values, mi, ma)  # scale between 0 and 1.
        else:
            for col in df.columns:  # go through all of the columns
                df[col] = normalize(df[col].values, coefs[col]["min"], coefs[col]["max"])  # scale with the given coeficients
        df.dropna(inplace=True)  # cleanup again... jic.

        sequential_data = []  # this is a list that will CONTAIN the sequences
        prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

        for i in df.values:  # iterate over the values
            prev_days.append([n for n in i[:-1]])  # store all but the target
            if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
                sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

        # Just shuffle if we need it
        if shuffle:
            random.shuffle(sequential_data)  # shuffle for good measure.

        X = []
        y = []

        for seq, target in sequential_data:  # going over our new sequential data
            X.append(seq)  # X is the sequences
            y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

        return np.array(X), y, coefs  # return X and y...and make X a numpy array!

    print(" - Reading csv.")
    raw_data = pd.read_csv("data/BTCUSDT.csv")
    print(" - Setting up csv.")
    raw_data.set_index("date", inplace=True)

    raw_data.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    raw_data.dropna(inplace=True)

    raw_data['target'] = raw_data['close'].shift(-FUTURE_PERIOD_PREDICT)

    raw_data.dropna(inplace=True)

    print(" - Creating train and validation datasets.")
    ## here, split away some slice of the future data from the main main_df.
    times = sorted(raw_data.index.values)
    last_5pct = sorted(raw_data.index.values)[-int(0.05*len(times))]

    validation_data = raw_data[(raw_data.index >= last_5pct)]
    raw_data = raw_data[(raw_data.index < last_5pct)]

    print(" - Process datasets")
    train_x, train_y, coefs = preprocess_df(raw_data)
    # Do not shuffle the validation for an easy plot
    validation_x, validation_y, _ = preprocess_df(validation_data, coefs, False)

    return train_x, train_y, validation_x, validation_y

def main():
    print("Loading dataset.")
    dataset = load_dataset()
    AISquared()

if __name__ == '__main__':
    print("Launching test script.")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()

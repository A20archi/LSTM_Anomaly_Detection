import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale(values):
    scaler = MinMaxScaler()
    return scaler.fit_transform(values)

def create_windows(data, window):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
    return np.array(X)

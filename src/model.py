from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

def build_model(window):
    model = Sequential([
        LSTM(64, input_shape=(window, 1), return_sequences=False),
        RepeatVector(window),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

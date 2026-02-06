import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model

from data_loader import load_data
from preprocess import scale, create_windows
from config import WINDOW_SIZE, THRESHOLD_STD

os.makedirs("results/plots", exist_ok=True)

df = load_data()
values = scale(df[['value']].values)

X = create_windows(values, WINDOW_SIZE)

# Ensure correct shape (safety check)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = load_model("models/lstm_autoencoder.h5", compile=False)
model.compile(optimizer="adam", loss="mse")


X_pred = model.predict(X)
recon_error = np.mean((X - X_pred) ** 2, axis=(1, 2))

threshold = recon_error.mean() + THRESHOLD_STD * recon_error.std()
anomalies = recon_error > threshold

# -------- PLOTS -------- #

# Raw signal
plt.figure(figsize=(15,4))
plt.plot(df['value'])
plt.title("Raw Time Series")
plt.savefig("results/plots/raw_signal.png")
plt.show()

# Reconstruction error
plt.figure(figsize=(15,4))
plt.plot(recon_error)
plt.axhline(threshold, color='red', linestyle='--')
plt.title("Reconstruction Error")
plt.savefig("results/plots/reconstruction_error.png")
plt.show()

# Detected anomalies
plt.figure(figsize=(15,4))
plt.plot(df['value'][WINDOW_SIZE:], label="signal")
plt.scatter(
    np.where(anomalies),
    df['value'][WINDOW_SIZE:][anomalies],
    color='red',
    label="anomaly"
)
plt.legend()
plt.title("Detected Anomalies")
plt.savefig("results/plots/detected_anomalies.png")
plt.show()

with open("results/summary.txt", "w") as f:
    f.write(f"Threshold used: mean + {THRESHOLD_STD} * std\n")
    f.write(f"Total detected anomalies: {anomalies.sum()}\n")

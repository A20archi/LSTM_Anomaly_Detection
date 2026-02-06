from data_loader import load_data
from preprocess import scale, create_windows
from model import build_model
from config import WINDOW_SIZE, EPOCHS, BATCH_SIZE
import matplotlib.pyplot as plt
import os

os.makedirs("models", exist_ok=True)

df = load_data()
values = scale(df[['value']].values)

X = create_windows(values, WINDOW_SIZE)

model = build_model(WINDOW_SIZE)

history = model.fit(
    X, X,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=False
)

model.save("models/lstm_autoencoder.h5")

plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="val")
plt.legend()
plt.savefig("results/plots/training_loss.png")
plt.show()

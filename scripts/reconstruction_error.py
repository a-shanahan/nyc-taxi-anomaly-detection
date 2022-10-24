import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

autoencoder = tf.keras.models.load_model('autoencoder_fraud.tf/')

with open('data/val_files.txt', 'r') as f:
    files = f.read().split("\n")

file = str(random.choice(files))

tmp = pd.read_parquet(file, engine='pyarrow')

dta = np.asarray(tmp).reshape(len(tmp), 1, 56).astype('float32')

logger.info(f'Chosen validation file: {file}')

encoded_data = autoencoder.encoder(dta).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(dta[0][0], 'b')
plt.plot(decoded_data[0][0], 'r')
plt.fill_between(np.arange(56), decoded_data[0][0], dta[0][0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title('Input vs Reconstruction')
plt.savefig('../graphs/InputReconstruction.png')

plt.clf()

reconstructions = autoencoder.predict(np.reshape(dta[0:2500], (2500, 1, 56)))
train_loss = tf.keras.losses.mae(reconstructions, np.reshape(dta[0:2500], (2500, 1, 56)))

# Calculate threshold as mean +1 std.
threshold = np.mean(train_loss) + np.std(train_loss)
# Reshape for plotting
x = tf.reshape(train_loss, [-1])

plt.hist(x, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.title(f'Threshold: {threshold}')
plt.savefig('../graphs/ErrorThreshold.png')

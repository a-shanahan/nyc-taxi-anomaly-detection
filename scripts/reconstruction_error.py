import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

autoencoder = tf.keras.models.load_model('autoencoder_fraud.tf/')

file = '/data/parDF.parquet/pickup_weekday=6/part-00002-63bfa723-e7d3-4d13-9320-40b06dbd7fc2.c000.snappy.parquet'

tmp = pd.read_parquet(file, engine='pyarrow')
dta = np.asarray(tmp).reshape(len(tmp), 1, 64).astype('float32')

encoded_data = autoencoder.encoder(dta).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(dta[0][0], 'b')
plt.plot(decoded_data[0][0], 'r')
plt.fill_between(np.arange(64), decoded_data[0][0], dta[0][0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title('Input vs Reconstruction')
plt.savefig('../graphs/InputReconstruction.png')

plt.clf()

reconstructions = autoencoder.predict(np.reshape(dta[0:2500], (2500, 1, 64)))
train_loss = tf.keras.losses.mae(reconstructions, np.reshape(dta[0:2500], (2500, 1, 64)))

# Calculate threshold as mean +1 std.
threshold = np.mean(train_loss) + np.std(train_loss)
# Reshape for plotting
x = tf.reshape(train_loss, [-1])

plt.hist(x, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.title(f'Threshold: {threshold}')
plt.savefig('../graphs/ErrorThreshold.png')

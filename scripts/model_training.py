import pandas as pd
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .autoencoder import *

# Check presence of GPU
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

file = '/data/parDF.parquet/pickup_weekday=1/part-00002-63bfa723-e7d3-4d13-9320-40b06dbd7fc2.c000.snappy.parquet'

df = pd.read_parquet(file, engine='pyarrow')
X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)

latent_dim = 8
autoencoder = Autoencoder(latent_dim)

cp = ModelCheckpoint(filepath="autoencoder_fraud.tf",
                     mode='min',
                     monitor='val_loss',
                     verbose=2,
                     save_format="tf",
                     save_best_only=True)

# define our early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = autoencoder.fit(X_train, X_train,
                          batch_size=128,
                          validation_data=(X_test, X_test),
                          callbacks=[cp, early_stop],
                          epochs=3).history

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

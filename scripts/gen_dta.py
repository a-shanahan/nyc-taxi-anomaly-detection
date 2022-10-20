import pandas as pd
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from autoencoder import *

# Check presence of GPU
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

root_dir = 'data/parDF.parquet'
files = []
for directory, subdirlist, filelist in os.walk(root_dir):
    for f in filelist:
        if re.search('parquet$', f):
            rel_dir = os.path.relpath(directory, root_dir)
            files.append(os.path.join(directory, f))

train_file_names, test_file_names = train_test_split(files, test_size=0.2, random_state=42)


def data_generator(file_list, batch_size=1):
    i = 0
    while True:
        if i * batch_size >= len(file_list):  # This loop is used to run the generator indefinitely.
            i = 0
            np.random.shuffle(file_list)
        else:
            file = file_list[i]
            tmp = pd.read_parquet(file.decode('utf-8'), engine='pyarrow')
            dta = np.asarray(tmp).reshape(len(tmp), 1, 64)
            yield dta, dta
            i = i + 1


batch_size = 1
train_dataset = tf.data.Dataset.from_generator(data_generator, args=[train_file_names, batch_size],
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=((None, 1, 64), (None, 1, 64)))

test_dataset = tf.data.Dataset.from_generator(data_generator, args=[test_file_names, batch_size],
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=((None, 1, 64), (None, 1, 64)))

latent_dim = 8
autoencoder = Autoencoder(latent_dim)

cp = ModelCheckpoint(filepath="autoencoder_fraud.tf",
                     mode='min',
                     monitor='val_loss',
                     verbose=2,
                     save_format="tf",
                     save_best_only=True)

# define early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = autoencoder.fit(train_dataset,
                          steps_per_epoch=len(train_file_names),
                          validation_steps=len(test_file_names),
                          batch_size=128,
                          validation_data=test_dataset,
                          callbacks=[cp, early_stop],
                          epochs=15).history

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

"""
This script trains the Autoencoder model using a Tensorflow data generator to stream Parquet files.
"""
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
# Walk data directory for parquet files
for directory, subdirlist, filelist in os.walk(root_dir):
    for f in filelist:
        if re.search('parquet$', f):
            rel_dir = os.path.relpath(directory, root_dir)
            files.append(os.path.join(directory, f))

# Split files into train test
train_file_names, test_file_names = train_test_split(files, test_size=0.2, random_state=42)

test_file_names, val_file_names = train_test_split(test_file_names, test_size=0.2, random_state=42)

with open('data/train_files.txt', 'w') as f:
    for line in train_file_names:
        f.write(f"{line}\n")

with open('data/test_files.txt', 'w') as f:
    for line in test_file_names:
        f.write(f"{line}\n")

with open('data/val_files.txt', 'w') as f:
    for line in val_file_names:
        f.write(f"{line}\n")


# Read in files as data generator as too large to fit in memeory
def data_generator(file_list, b_size=1):
    i = 0
    while True:
        if i * b_size >= len(file_list):
            i = 0
            np.random.shuffle(file_list)
        else:
            file = file_list[i]
            tmp = pd.read_parquet(file.decode('utf-8'), engine='pyarrow')
            dta = np.asarray(tmp).reshape(len(tmp), 1, 56)
            yield dta, dta
            i = i + 1


# Create tensorflow datasets using generator
batch_size = 1
train_dataset = tf.data.Dataset.from_generator(data_generator, args=[train_file_names, batch_size],
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=((None, 1, 56), (None, 1, 56)))

test_dataset = tf.data.Dataset.from_generator(data_generator, args=[test_file_names, batch_size],
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=((None, 1, 56), (None, 1, 56)))

latent_dim = 8
autoencoder = Autoencoder(latent_dim)

# Define model save params
cp = ModelCheckpoint(filepath="autoencoder_fraud.tf",
                     mode='min',
                     monitor='val_loss',
                     verbose=2,
                     save_format="tf",
                     save_best_only=True)

# Define early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=3,
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

# Plot training loss
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('../graphs/TrainingLoss.png')

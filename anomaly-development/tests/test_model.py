import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import pytest

module_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(module_dir, '../scripts/'))

from scripts.autoencoder import Autoencoder


def test_autoencoder():
    """
    Checks if output layer of model is of the correct shape
    :return:
    """
    with open('../anomaly-development/scripts/data/val_files.txt', 'r') as f:
        val_file_names = f.read().splitlines()
    file = os.path.join('../anomaly-development/scripts', val_file_names[0])
    tmp = pd.read_parquet(file, engine='pyarrow')
    dta = np.asarray(tmp).reshape(len(tmp), 1, 56)
    tf_data = tf.convert_to_tensor(dta[0], dtype=tf.float32)
    tf_data = tf.reshape(tf_data, (1, 1, 56))
    expected_shape = (1, 1, 56)
    model = Autoencoder(8)
    assert expected_shape == model.call(tf_data).shape


test_autoencoder()

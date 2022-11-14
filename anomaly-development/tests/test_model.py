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
    expected_shape = (1, 1, 56)
    input_data = tf.zeros(expected_shape,
                          dtype=tf.dtypes.float32)
    model = Autoencoder(8)
    assert expected_shape == model.call(input_data).shape


test_autoencoder()

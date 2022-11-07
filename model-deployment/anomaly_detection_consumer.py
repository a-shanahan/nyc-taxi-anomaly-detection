"""
This script monitors the 'completed-journeys' Kafka topic for completed taxi journeys.
The data has already been processed into the required format so can be passed directly
to the trained autoencoder model which is loaded from another directory. The predicted label
is used to determine which Kafka topic is subsequently notified.
"""
import logging
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
import tensorflow as tf
import numpy as np
import json
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

autoencoder = tf.keras.models.load_model('../anomaly-development/scripts/autoencoder_fraud.tf/')


def anomaly_detection(data: Dict, threshold: float = 0.074) -> Any:
    """
    Determine if journey profile is anomalous
    :param data: Journey data
    :param threshold: Anomaly threshold
    :return: Anomalous/Normal flag
    """
    logger.info(f'uid: {data["uid"]}')
    # Remove UID from journey profile as model has not been trained on this
    data.pop('uid')
    model_input = np.asarray([value for key, value in data.items()])
    try:
        dta = np.asarray(model_input).reshape(1, 1, 56).astype('float32')
        reconstructions = autoencoder.predict(dta)
        train_loss = tf.keras.losses.mae(reconstructions, dta)
        if train_loss > threshold:
            return True
        else:
            return False
    except ValueError as e:
        logger.debug(f'Length error: {e}')
        return False


def start_consumer():
    while True:
        mess = consumer.poll(0.1, max_records=1)
        if len(mess) == 0:
            continue
        try:
            for _, j in mess.items():
                for message in j:
                    message_decode = json.loads(message.value.decode())
                    uid = message_decode['uid']
                    prediction = anomaly_detection(message_decode)
                    # Send notification to either normal or anomalous topic
                    if prediction:
                        ack = producer.send('anomalous-journey', uid.encode('utf-8'))
                        _ = ack.get()
                    else:
                        ack = producer.send('normal-journey', uid.encode('utf-8'))
                        _ = ack.get()
        except NoBrokersAvailable as e:
            logger.debug(f"Error: {e}")
            pass


if __name__ == '__main__':
    consumer = KafkaConsumer(bootstrap_servers='localhost:29092')
    producer = KafkaProducer(bootstrap_servers='localhost:29092')
    consumer.subscribe(['journey-finished'])
    start_consumer()

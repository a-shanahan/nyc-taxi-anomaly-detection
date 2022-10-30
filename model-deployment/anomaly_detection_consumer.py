import logging
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
import tensorflow as tf
import numpy as np
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

autoencoder = tf.keras.models.load_model('../anomaly-development/scripts/autoencoder_fraud.tf/')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def anomaly_detection(data, threshold=0.074):
    logger.info(f'uid: {data["uid"]}')
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

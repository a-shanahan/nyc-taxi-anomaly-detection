import logging
import argparse
import json
import configparser
from typing import Dict
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
from stable_baselines3 import PPO
from utilities.assigner_utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

configParser = configparser.ConfigParser()
configFilePath = "data/db_config.ini"
configParser.read(configFilePath)

db_config = {'user': configParser['MariaDB']['user'],
             'password': configParser['MariaDB']['password'],
             'host': configParser['MariaDB']['host'],
             'database': configParser['MariaDB']['database']}

assigner = AssignerUtils(db_config)


def producer_format(topic: str, message: Dict):
    """
    Send message to Kafka topic
    :param topic: Kafka Topic to send message
    :param message: JSON formatted message
    """
    ack = producer.send(topic, json.dumps(message).encode('utf-8'))
    _ = ack.get()


drivers = {}
increment = 0.5


def start_holding_pool():
    while True:
        mess = consumer.poll(increment, max_records=1)
        if len(mess) == 0:
            continue
        try:
            for _, j in mess.items():
                for message in j:
                    order = json.loads(message.value.decode('utf-8'))
                    # Calculate journey time
                    journey_time = pd.to_datetime(order['order']['tpep_dropoff_datetime']) - \
                                   pd.to_datetime(order['order']['tpep_pickup_datetime'])
                    # Create dictionary to store driver details
                    drivers.update({order['driver']: {
                        'journey_time': journey_time.seconds/60,  # TODO: Change back to seconds
                        'order': order['order']}})
                    remove_drivers = []
                    # Iterate through driver dictionary and if
                    for key, value in drivers.items():
                        drivers[key]['journey_time'] = drivers[key]['journey_time'] - increment
                        if drivers[key]['journey_time'] <= 0:
                            logger.info(f'Driver finished: {key}')
                            # Send journey details to Spark for processing
                            producer_format('completed-journey', drivers[key]['order'])
                            # Change driver status to available
                            msg = {'driver': key,
                                   'status': 'Y',
                                   'order': drivers[key]['order']}
                            producer_format('driver-status', msg)
                            # Add driver to list for removal
                            remove_drivers.append(key)
                    for driver in remove_drivers:
                        del drivers[driver]
        except NoBrokersAvailable as e:
            logger.debug(f"Error: {e}")
            pass


if __name__ == '__main__':
    consumer = KafkaConsumer(bootstrap_servers='localhost:29092', api_version=(0, 10, 1))
    producer = KafkaProducer(bootstrap_servers='localhost:29092', api_version=(0, 10, 1))
    consumer.subscribe('hold-driver')
    start_holding_pool()

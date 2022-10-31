import logging
import argparse
import json
import configparser
from typing import Dict
import numpy as np
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
                    logger.info(f'Topic: "hold-driver" Value: {order}')
                    drivers.update({order['driver']: {
                                        'time_delta': order['time_delta'],
                                        'order': order['order']}})
                    assigner.driver_assignment('N', order['driver'])
                    remove_drivers = []
                    for key, value in drivers.items():
                        drivers[key]['time_delta'] = drivers[key]['time_delta'] - increment
                        if drivers[key]['time_delta'] <= 0:
                            producer_format('journey-finished', drivers[key]['order'])
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

import logging
import argparse
import json
import configparser
import random
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

model = PPO.load("../assignment_development/model/taxi-assigner")

assigner = AssignerUtils(db_config)


def producer_format(topic: str, message: Dict):
    """
    Send message to Kafka topic
    :param topic: Kafka Topic to send message
    :param message: JSON formatted message
    """
    ack = producer.send(topic, json.dumps(message).encode('utf-8'))
    _ = ack.get()


def start_decider():
    while True:
        mess = consumer.poll(0.1, max_records=1)
        if len(mess) == 0:
            continue
        try:
            for _, j in mess.items():
                for message in j:
                    driver = None
                    time_delta = None
                    order = json.loads(message.value.decode('utf-8'))
                    logger.info(f'Topic: "new-order" Value: {order}')
                    customer_order = {"PULocation": order['PULocationID'],
                                      "DOLocation": order['DOLocationID'],
                                      "pickup_time": random.randint(5, 30),
                                      "fare": order['fare_amount']}
                    # Generate current environment observation and choose driver
                    obs = assigner.customer_order(customer_order)
                    obs = np.reshape(obs, (1, 8))
                    action_type = model.predict(obs)[0]
                    refuse = False
                    if action_type < -0.5:
                        driver, time_delta = assigner.refuse()
                    elif -0.5 < action_type < 0:
                        driver, time_delta = assigner.closest_driver_assignment()
                    elif 0 < action_type < 0.5:
                        driver, time_delta = assigner.lowest_utilisation_driver_assignment()
                    elif action_type > 0.5:
                        driver, time_delta = assigner.random_driver_assignment()

                    if refuse:
                        msg = {'uid': order['uid'],
                               'Response': 'refused',
                               'Message': 'Sorry we are unable to take your order'}
                        producer_format('customer-response', msg)
                    elif not driver:
                        msg = {'uid': order['uid'],
                               'Response': 'busy',
                               'Message': 'Sorry all our drivers are currently busy'}
                        producer_format('customer-response', msg)
                    else:
                        if time_delta < 0:
                            resp = f'Your driver {driver} will be {abs(time_delta)} minutes late'
                        else:
                            resp = f'Your driver {driver} will be along in {time_delta} minutes'
                        msg = {'uid': order['uid'],
                               'Response': 'accepted',
                               'Message': resp}
                        # Notify customer
                        producer_format('customer-response', msg)
                        # Hold driver for period of time before releasing journey details. Not needed in
                        # production system
                        hold_msg = {'order': order,
                                    'driver': driver,
                                    'time_delta': time_delta}
                        producer_format('hold-driver', hold_msg)
        except NoBrokersAvailable as e:
            logger.debug(f"Error: {e}")
            pass


if __name__ == '__main__':
    consumer = KafkaConsumer(bootstrap_servers='localhost:29092', api_version=(0, 10, 1))
    producer = KafkaProducer(bootstrap_servers='localhost:29092', api_version=(0, 10, 1))
    consumer.subscribe('new-order')
    start_decider()

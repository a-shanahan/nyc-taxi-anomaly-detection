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

model = PPO.load("../assignment_development/model/taxi-assigner")

assigner = AssignerUtils(db_config)


def producer(topic: str, message: Dict):
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
                    logger.info(f'Topic: "new-order" Value: {message.value}')
                    customer_order = {"PULocation": message.value['PULocation'],
                                      "DOLocation": message.value['DOLocation'],
                                      "pickup_time": message.value['pickup_time'],
                                      "fare": message.value['fare']}
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
                        msg = {'uid': message.value['uid'],
                            'Response': 'refused',
                               'Message': 'Sorry we are unable to take your order'}
                        producer('customer_response', msg)
                    elif not driver:
                        msg = {'uid': message.value['uid'],
                            'Response': 'busy',
                               'Message': 'Sorry all our drivers are currently busy'}
                        producer('customer_response', msg)
                    else:
                        msg = {'uid': message.value['uid'],
                            'Response': 'accepted',
                               'Message': f'Your driver {driver} will be along in {time_delta} minutes'}
                        producer('customer_response', msg)
        except NoBrokersAvailable as e:
            logger.debug(f"Error: {e}")
            pass


if __name__ == '__main__':
    consumer = KafkaConsumer(bootstrap_servers='localhost:29092')
    producer = KafkaProducer(bootstrap_servers='localhost:29092')
    consumer.subscribe('new-order')
    start_decider()

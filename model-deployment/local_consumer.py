import logging
import argparse
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def start_consumer(topic):
    while True:
        mess = consumer.poll(0.1, max_records=1)
        if len(mess) == 0:
            continue
        try:
            for _, j in mess.items():
                for message in j:
                    logger.info(f'Topic: {topic} Value: {message.value}')
        except NoBrokersAvailable as e:
            logger.debug(f"Error: {e}")
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')
    args = parser.parse_args()

    consumer = KafkaConsumer(bootstrap_servers='localhost:29092')
    consumer.subscribe([args.topic])
    start_consumer(args.topic)

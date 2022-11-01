import logging
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable
import mysql.connector as connector
from sqlalchemy import create_engine
import json
import configparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load in MAriaDB configuration

configParser = configparser.ConfigParser()
configFilePath = "data/db_config.ini"
configParser.read(configFilePath)

db_config = {'user': configParser['MariaDB']['user'],
             'password': configParser['MariaDB']['password'],
             'host': configParser['MariaDB']['host'],
             'database': configParser['MariaDB']['database']}

# Connect to MariaDB
conn = connector.connect(host=db_config['host'],
                         user=db_config['user'],
                         password=db_config['password'],
                         database=db_config['database'])
uri = f"mysql+mysqlconnector://{db_config.get('user')}:{db_config.get('password')}@" \
      f"{db_config.get('host')}/{db_config.get('database')}"
engine = create_engine(uri)
# Get Cursor
cursor = conn.cursor()


def query_execute(query: str) -> None:
    """
    Execute query on database
    :param query: SQL query
    """
    cursor.execute(query)


def start_consumer():
    while True:
        mess = consumer.poll(0.1, max_records=1)
        if len(mess) == 0:
            continue
        try:
            for _, j in mess.items():
                for message in j:
                    msg = json.loads(message.value.decode())
                    logger.info(f'Message: {msg}')
                    # Update driver availability
                    query_execute("UPDATE availability SET Available = '" + msg['status'] +
                                  "' WHERE Driver = '" + msg['driver'] + "'")

                    # Drivers only become available when journey is completed
                    # If the driver has just completed a journey, update their stats
                    if msg['status'] == 'Y':
                        query_execute("UPDATE stats SET Total_Fare = "
                                      "Total_Fare + '" + str(msg['order'].get("fare_amount")) +
                                      "' WHERE Driver = '" + msg['driver'] + "'")
                        query_execute("UPDATE availability SET Location = '" +
                                      str(msg['order'].get('DOLocationID')) +
                                      "' WHERE Driver = '" + msg['driver'] + "'")
        except NoBrokersAvailable as e:
            logger.debug(f"Error: {e}")
            pass


if __name__ == '__main__':
    consumer = KafkaConsumer(bootstrap_servers='localhost:29092')
    producer = KafkaProducer(bootstrap_servers='localhost:29092')
    consumer.subscribe(['driver-status'])
    start_consumer()

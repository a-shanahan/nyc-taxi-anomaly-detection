"""
This script reads data from a nyc taxi parquet file and produces messages to the 'new-order' Kafka topic
for taxi driver assignment. Time deltas between orders are calculated and this value is used to determine
how long the for loop should wait between publishing a new message.
"""
import pandas as pd
import time
from kafka import KafkaProducer
import uuid
import json

producer = KafkaProducer(bootstrap_servers='localhost:29092', api_version=(0, 10, 1))

file = '../anomaly-development/scripts/data/raw_data/yellow_tripdata_2019-01.parquet'

locations = pd.read_csv('../assignment_development/data/taxi_cords.csv')

tmp = pd.read_parquet(file, engine='pyarrow')

tmp = tmp[tmp['PULocationID'].isin(locations.LocationID)]
tmp = tmp[tmp['DOLocationID'].isin(locations.LocationID)]

tmp['tpep_pickup_datetime'] = pd.to_datetime(tmp['tpep_pickup_datetime'])
tmp = tmp[tmp['tpep_pickup_datetime'] > '2019-01-01']
tmp.sort_values('tpep_pickup_datetime', ascending=True, inplace=True)
# Compute time_delta
tmp['order_delta'] = pd.to_datetime(tmp['tpep_pickup_datetime']) - pd.to_datetime(tmp['tpep_pickup_datetime']).shift()
tmp['order_delta'] = tmp['order_delta'].dt.seconds

tmp['tpep_pickup_datetime'] = tmp['tpep_pickup_datetime'].astype(str)
tmp['tpep_dropoff_datetime'] = tmp['tpep_dropoff_datetime'].astype(str)

for index, row in tmp.iterrows():
    msg = row.to_json(orient='index')
    msg = json.loads(msg)
    _wait = msg['order_delta']
    del msg['order_delta']
    msg.update({'uid': uuid.uuid4().hex})
    ack = producer.send('new-order', json.dumps(msg).encode('utf-8'))
    metadata = ack.get()
    try:
        time.sleep(_wait)
    except TypeError:
        pass

import pandas as pd
import time
import random
from kafka import KafkaProducer
import uuid
import json

producer = KafkaProducer(bootstrap_servers='localhost:29092')

file = '../anomaly-development/scripts/data/raw_data/yellow_tripdata_2019-01.parquet'

tmp = pd.read_parquet(file, engine='pyarrow')
tmp.sort_values('tpep_pickup_datetime', ascending=True, inplace=True)
# Compute time_delta
tmp['time_delta'] = tmp['tpep_pickup_datetime'] - tmp['tpep_pickup_datetime'].shift()
tmp['tpep_pickup_datetime'] = tmp['tpep_pickup_datetime'].astype(str)
tmp['tpep_dropoff_datetime'] = tmp['tpep_dropoff_datetime'].astype(str)

for index, row in tmp.iterrows():
    msg = row.to_json(orient='index')
    msg = json.loads(msg)
    msg.update({'uid': uuid.uuid4().hex})
    ack = producer.send('new-order', json.dumps(msg).encode('utf-8'))
    metadata = ack.get()
    time.sleep(msg['time_delta'])

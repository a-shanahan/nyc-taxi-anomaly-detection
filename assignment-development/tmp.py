import logging
import random
import pandas as pd


tmp = pd.read_parquet('/Users/alexshanahan/Documents/GitHub/nyc-taxi-anomaly-detection/anomaly-development/scripts/data/raw_data/yellow_tripdata_2019-01.parquet')
print(tmp.columns)
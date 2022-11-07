"""
This script generates synthetic data for use training the assigner model.
"""
import pandas as pd
import uuid
import random
from random import randrange
import datetime

number = 30
df = pd.DataFrame({'Driver': [uuid.uuid4().hex for i in range(number)],
                   'Total_Free': [random.randint(0, 120) for i in range(number)],
                   'Total_Busy': [random.randint(0, 120) for i in range(number)],
                   'Total_Fare': [random.randint(5, 100) for i in range(number)],
                   'Total_Journeys': [random.randint(15, 20) for i in range(number)],
                   'Total_Distance': [random.randint(50, 100) for i in range(number)]})

df['Utilisation'] = df['Total_Fare'].apply(lambda x: x / max(df.Total_Fare))
df.set_index(keys='Driver', inplace=True)
df.to_csv('data/df_stats.csv')


def random_date(start, l):
    current = start
    while l >= 0:
        current = current + datetime.timedelta(minutes=randrange(10))
        yield current
        l -= 1


startDate = datetime.datetime(2013, 9, 20, 8, 00)

locs = pd.read_csv('data/taxi_cords.csv')
locations = list(locs.LocationID)

driver_availability = pd.DataFrame({'Location': [random.choice(locations) for i in range(number)],
                                    'Driver': df.index,
                                    'Available': [random.choice(['Y', 'N']) for i in range(number)],
                                    'Time_until_free': [random.randint(1, 10) for i in range(number)]})

driver_availability['Time'] = [x.strftime("%H:%M") for x in reversed(list(random_date(startDate, (number - 1))))]

driver_availability.loc[driver_availability['Available'] == 'N', 'Time'] = None
driver_availability.loc[driver_availability['Available'] == 'Y', 'Time_until_free'] = 0
driver_availability.set_index(keys='Driver', inplace=True)

driver_availability.to_csv('data/driver_availability.csv')

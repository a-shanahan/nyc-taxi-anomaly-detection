import pandas as pd
import uuid
import random
from random import randrange
import datetime
from itertools import combinations_with_replacement

number = 100

df = pd.DataFrame({'Driver': [uuid.uuid4().hex for i in range(number)],
                   'Total_Free': [random.randint(0, 120) for i in range(number)],
                   'Total_Busy': [random.randint(0, 120) for i in range(number)],
                   'Total_Fare': [random.randint(0, 100) for i in range(number)],
                   'Total_Journeys': [random.randint(0, 25) for i in range(number)],
                   'Total_Distance': [random.randint(0, 100) for i in range(number)]})

df['Utilisation'] = (df['Total_Busy'] / df['Total_Free'])
df.set_index(keys='Driver', inplace=True)
df.to_csv('data/df_stats.csv')


def random_date(start, l):
    current = start
    while l >= 0:
        current = current + datetime.timedelta(minutes=randrange(10))
        yield current
        l -= 1


startDate = datetime.datetime(2013, 9, 20, 8, 00)

locations = ['a', 'b', 'c', 'd']

driver_availability = pd.DataFrame({'Location': [random.choice(locations) for i in range(number)],
                                    'Driver': df.index,
                                    'Available': [random.choice(['Y', 'N']) for i in range(number)],
                                    'Time_until_free': [random.randint(1, 10) for i in range(number)]})

driver_availability['Time'] = [x.strftime("%H:%M") for x in reversed(list(random_date(startDate, (number - 1))))]

driver_availability.loc[driver_availability['Available'] == 'N', 'Time'] = None
driver_availability.loc[driver_availability['Available'] == 'Y', 'Time_until_free'] = 0
driver_availability.set_index(keys='Driver', inplace=True)

driver_availability.to_csv('data/driver_availability.csv', index=False)


sample_list = ['a', 'b', 'c', 'd']
journeys = list(combinations_with_replacement(sample_list, 2))
journeys_2 = list(combinations_with_replacement(reversed(sample_list), 2))
journeys = journeys + journeys_2
journeys = list(set(journeys))

lookup = pd.DataFrame({'journeys': journeys,
                       'time': [random.randint(1, 10) for i in range(len(journeys))]})

lookup['journeys'] = lookup['journeys'].astype(str)
lookup['start'] = lookup['journeys'].str[2:3]
lookup['finish'] = lookup['journeys'].str[7:8]

lookup.drop(columns='journeys', inplace=True)
lookup.to_csv('data/lookup.csv', index=False)

from assigner_utils import *
from stable_baselines3 import PPO
import numpy as np
import configparser

configParser = configparser.ConfigParser()
configFilePath = "data/db_config.txt"
configParser.read(configFilePath)

db_config = {'user': configParser['MariaDB']['user'],
             'password': configParser['MariaDB']['password'],
             'host': configParser['MariaDB']['host'],
             'database': configParser['MariaDB']['database']}

customer_order = {"PULocation": '1',
                  "DOLocation": '2',
                  "pickup_time": 10,
                  "fare": '100'}

model = PPO.load("../assignment_development/model/taxi-assigner")

assigner = AssignerUtils(db_config)

# Generate order
obs = assigner.customer_order(customer_order)
obs = np.reshape(obs, (1, 8))
action_type = model.predict(obs)[0]
print(f'Action: {action_type[0]}')

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
    print("Sorry, we're too busy atm")
elif not driver:
    print("Sorry, we don't have anyone to take your order")
else:
    print(f'Driver: {driver} TimeDelta: {time_delta}')

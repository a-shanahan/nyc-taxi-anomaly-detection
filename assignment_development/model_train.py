import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from assigner import *

availability = 'data/driver_availability.csv'
stats = 'data/df_stats.csv'
locations = 'data/taxi_cords.csv'

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: AssignmentEnv(availability, stats, locations)])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./delivery_assignment_tensorboard/", device='mps')
model.learn(total_timesteps=5000)

# save the model
model.save("model/taxi-assigner")
policy = model.policy
policy.save("model/taxi-assigner-policy.pkl")

"""
This script evaluates the assigner model by running a number of simulations and comparing the output
to a number of naive choices.
"""
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from assigner import *

rl_agent = []
closest_driver = []
even_utilisation = []
any_available = []
random_choice = []
actions = []

driver_availability = 'data/driver_availability.csv'
df_stats = 'data/df_stats.csv'
locs = 'data/taxi_cords.csv'

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: AssignmentEnv(driver_availability, df_stats, locs)])

model = PPO.load("model/taxi-assigner", env=env)

for j in range(20):
    print(f'Step: {j}')
    # RL Agent
    obs = env.reset()
    while True:
        tmp = env.render()
        action, _states = model.predict(obs)
        actions.append(str(action))
        obs, rewards, done, info = env.step(action)
        if done:
            rl_agent.append(tmp)
            break

    # # Closest driver
    obs = env.reset()
    while True:
        tmp = env.render()
        obs, rewards, done, info = env.step(np.array([[-0.25]]))
        if done:
            closest_driver.append(tmp)
            break

    # Utilisation
    obs = env.reset()
    while True:
        tmp = env.render()
        obs, rewards, done, info = env.step(np.array([[0.25]]))
        if done:
            even_utilisation.append(tmp)
            break

    # Any available
    obs = env.reset()
    while True:
        tmp = env.render()
        obs, rewards, done, info = env.step(np.array([[1]]))
        if done:
            any_available.append(tmp)
            break

    # Random action
    obs = env.reset()
    while True:
        tmp = env.render()
        obs, rewards, done, info = env.step(np.array([env.action_space.sample()]))
        if done:
            random_choice.append(tmp)
            break

df_rl = pd.DataFrame(rl_agent).describe()
df_util = pd.DataFrame(even_utilisation).describe()
df_closest = pd.DataFrame(closest_driver).describe()
df_any = pd.DataFrame(any_available).describe()
df_random = pd.DataFrame(random_choice).describe()

df_rl.to_csv('data/rl.csv')
df_util.to_csv('data/util.csv')
df_closest.to_csv('data/close.csv')
df_any.to_csv('data/any.csv')
df_random.to_csv('data/random.csv')

with open('data/actions.txt', 'w') as f:
    f.writelines(actions)

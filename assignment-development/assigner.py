from pyparsing.helpers import DebugStartAction
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from copy import copy
from math import radians, cos, sin, asin, sqrt


class AssignmentEnv(gym.Env):
    """A Taxi Driver assignment environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df_availability, df_stats, taxi_zones):
        # Take copies of the imported datasets so that the environment can be reset
        self.availability_reset = df_availability.copy()
        self.df_availability = self.availability_reset

        self.stats_reset = df_stats.copy()
        self.df_stats = self.stats_reset

        # self.journeys = journeys.copy()

        self.taxi_zones = taxi_zones

        # self.locations = ['a', 'b', 'c', 'd']

        self.time_increment = 1  # New customer order occurs every 1 min

        # Initialise variables
        self.order = None
        self.driver = None
        self.total_reward = 0

        # Environment start/stop conditions
        self.total_allowable_loss = -100
        self.max_profit = 1000

        # Current state of the environment
        self.starting_available_drivers = copy(
            self.df_availability[self.df_availability.Available == 'Y']["Available"].count())
        self.available_drivers = self.starting_available_drivers

        self.fulfilled_order = 20

        self.avg_wait_time = 5

        self.dropped_order = 0
        self.rejected_order = 0

        self.profit = 0
        self.total_orders = 20  # Starting total orders

        # Penalise the model if the spread of total earnings isn't evenly distributed
        self.starting_fare_spread = self.df_stats.Total_Fare.max() - self.df_stats.Total_Fare.min()
        self.fare_spread = self.starting_fare_spread

        # Penalise the model if drivers are being under utilised
        self.starting_utilisation = len(self.df_stats[self.df_stats.Utilisation < 1])
        self.spread = self.starting_utilisation

        # Penalise the model if an order is rejected
        self.dropped_order_penalty = -10

        # Initial driver utilisation counts
        self.start_driver_used_counter = self.df_stats.Total_Journeys.to_dict().copy()
        self.driver_used_counter = self.start_driver_used_counter.copy()

        self.pickup_durations = np.array([])  # Array to store the collection wait times

        # Action space to choose the assignment algorithm
        # Most algorithms require a normalised action space
        self.action_space = spaces.Box(
            low=np.array([-1]), high=np.array([1]), shape=(1,), dtype=np.float16)

        # Observation space contains the business KPIs
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 7), dtype=np.float16)

    def _haversine(self, PULocation, DOLocation, speed=20):
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)
        """
        lon1 = self.taxi_zones[self.taxi_zones.LocationID == PULocation]['Longitude'].iloc[0]
        lat1 = self.taxi_zones[self.taxi_zones.LocationID == PULocation]['Latitude'].iloc[0]

        lon2 = self.taxi_zones[self.taxi_zones.LocationID == DOLocation]['Longitude'].iloc[0]
        lat2 = self.taxi_zones[self.taxi_zones.LocationID == DOLocation]['Latitude'].iloc[0]

        # convert decimal degrees to radians

        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return (c * r) / speed

    def _next_observation(self):
        """Generate a random customer order every timestep and store it as
        environment observations. These are normalised to be between 0-1."""

        self._customer_order()  # Generate customer order
        # Identify available drivers
        self.available_drivers = self.df_availability[self.df_availability.Available == 'Y']["Available"].count()

        # Create obs and normalise to between 0-1
        obs = np.array([
            self.fulfilled_order / self.total_orders,  # Fulfilled order %
            self.order.get("pickup_time") / 20,  # Max pickup time is 20 mins
            self.order.get("dropoff_time") / (30 + self.order.get("pickup_time")),
            self.dropped_order / self.total_orders,  # Dropped order %,
            self.rejected_order / self.total_orders,  # Dropped order %,
            self.spread,  # Spread of driver utilisation
            self.available_drivers / len(self.df_availability)])  # No. of available drivers
        return obs

    def _customer_order(self):
        """Build a customer order"""
        self.order = {"pickup_location": random.choice(self.taxi_zones.LocationID)}
        self.order.update({"dropoff_location": random.choice(self.taxi_zones.LocationID)})
        self.order.update(
            {"pickup_time": random.randint(5, 20)})  # Orders are ready to be picked up between 5 and 20 mins

        # drop_offtime = self.journeys[(self.journeys.start == self.order['pickup_location']) & (
        #             self.journeys.finish == self.order['dropoff_location'])]

        duration = self._haversine(self.order['pickup_location'], self.order['dropoff_location'])

        self.order.update({"dropoff_time": duration})  # Estimated Duration

        self.order.update({"fare": random.randint(3, 20)})

    def _update_driver_availability(self):
        """Update the driver availability lookup.
        This is to be run before every time step """
        self.df_availability.Time_until_free = self.df_availability.Time_until_free - self.time_increment
        self.df_availability.loc[(self.df_availability.Time_until_free <= 0), ['Time_until_free', "Available"]] = [0,
                                                                                                                   'Y']

    def _driver_assignment(self):
        """Once an order is assigned to a driver change their availability status.
        This is to be run after every time step."""

        self.df_availability.at[self.driver, 'Available'] = "N"
        # Sometimes a driver gets stuck in traffic and is out of commission for longer than expected
        if np.random.randint(0, 10) == 9:
            self.df_availability.at[self.driver, 'Time_until_free'] = self.order.get("dropoff_time") + 30
        else:
            self.df_availability.at[self.driver, 'Time_until_free'] = self.order.get("dropoff_time")

        # Increment their number of assigned deliveries
        self.df_stats.at[self.driver, 'Total_Journeys'] = self.df_stats.at[self.driver, 'Total_Journeys'] + 1

    def _random_driver_assignment(self, order_location):
        """This algorithm picks any available driver"""

        try:
            self.driver = self.df_availability[self.df_availability.Available == 'Y'].sample(1).index[0]
            driver_location = self.df_availability.loc[self.driver]['Location']

            duration = self._haversine(driver_location, self.order['dropoff_location'])

            # Extra reward given if order is picked up before estimated time
            time_delta = self.order.get("pickup_time") - duration
            self.fulfilled_order += 1

            if time_delta < 0:
                return self.order.get("fare") - abs(time_delta), time_delta
            else:
                return self.order.get("fare"), time_delta

        except ValueError:
            # If no driver is available then the order is treated as dropped
            self.dropped_order += 1
            return None, None

    def _closest_driver_assignment(self, order_location):
        """This algorithm picks the driver with the quickest lead time to pickup"""

        drivers = list(self.df_availability[self.df_availability.Available == 'Y'].index)
        if len(drivers) > 0:
            min_duration = 0
            choice = ''
            for index, driver in enumerate(drivers):
                driver_location = self.df_availability.loc[driver]['Location']

                duration = self._haversine(driver_location, self.order['dropoff_location'])

                if index == 0:
                    min_duration = duration
                    self.driver = driver
                elif duration < min_duration:
                    min_duration = duration
                    self.driver = driver

            order_pickup_duration = min_duration
            # Extra reward given if order is picked up before estimated time

            time_delta = self.order.get("pickup_time") - order_pickup_duration
            self.fulfilled_order += 1
            if time_delta < 0:
                return self.order.get("fare") - abs(time_delta), time_delta
            else:
                return self.order.get("fare"), time_delta

        else:
            # If no driver is available then the order is treated as dropped
            self.dropped_order += 1
            return None, None

    def _lowest_utilisation_driver_assignment(self, order_location):
        """This algorithm picks the driver with the lowest utilisation figure"""

        drivers = list(self.df_availability[self.df_availability.Available == 'Y'].index)
        if len(drivers) > 0:
            lowest_utilisation = self.df_stats[self.df_stats.index.isin(drivers)]["Utilisation"].min()
            self.driver = self.df_stats[self.df_stats["Utilisation"] == lowest_utilisation].index[0]

            driver_location = self.df_availability.loc[self.driver]['Location']

            duration = self._haversine(driver_location, self.order['dropoff_location'])

            # Reduce reward for late pickups
            time_delta = self.order.get("pickup_time") - duration

            self.fulfilled_order += 1

            if time_delta < 0:
                return self.order.get("fare") - abs(time_delta), time_delta
            else:
                return self.order.get("fare"), time_delta
        else:
            # If no driver is available then the order is treated as dropped
            self.dropped_order += 1
            return None, None

    def _refuse_order(self):
        """The model has the choice to reject the order before it is too late"""
        self.rejected_order += 1
        return -1, True

    def _take_action(self, action):
        action_type = action[0]
        # Reset variables
        refuse = False
        reward = 0
        order_pickup_duration = 0

        if action_type < -0.5:
            reward, refuse = self._refuse_order()

        elif -0.5 < action_type < 0:
            reward, order_pickup_duration = self._closest_driver_assignment(self.order.get("pickup_location"))

        elif 0 < action_type < 0.5:
            reward, order_pickup_duration = self._lowest_utilisation_driver_assignment(
                self.order.get("pickup_location"))

        elif action_type > 0.5:
            reward, order_pickup_duration = self._random_driver_assignment(self.order.get("pickup_location"))

        # Update driver order numbers
        if reward is not None and refuse is False:
            # If order is picked up then assigned driver needs delivery counter incremented by 1
            busy_counter = self.driver_used_counter.get(self.driver)
            self.driver_used_counter.update({self.driver: busy_counter + 1})

        # Update driver utilisation stats
        driver_utilisation = {key: (value / self.total_orders) for key, value in self.driver_used_counter.items()}
        utilisation_df = pd.DataFrame.from_dict(driver_utilisation, orient='index').rename(columns={0: "Utilisation"})
        self.df_stats.update(utilisation_df)

        # Calculate spread of utilisation figures
        self.spread = len(self.df_stats[self.df_stats.Utilisation < 1])

        if order_pickup_duration is not None and refuse is False:
            self.pickup_durations = np.append(self.pickup_durations, order_pickup_duration)
            self.avg_pickup_time = self.pickup_durations.mean()

            # Update driver status
            self._driver_assignment()

            # Reduce reward if more than 10% of drivers have low utilisation
            if ((self.spread / len(self.df_stats)) > 0.1) or (self.avg_pickup_time < 5):
                reward = self.order.get("fare") * 0.7

        if reward is not None:
            return reward
        else:
            # Order is unfulfilled
            return self.dropped_order_penalty

    def step(self, action):
        self._update_driver_availability()  # Update current driver availability
        self.total_reward = self._take_action(action)  # Choose assignment algorithm

        self.profit += self.total_reward  # Calculate running profit
        self.total_orders += 1  # Increment total orders
        if self.profit <= self.total_allowable_loss or self.profit >= self.max_profit:
            done = True  # If loss is too great  stop simulation
        else:
            done = False
        obs = self._next_observation()  # Update environment state

        return obs, self.total_reward, done, {"Agent action": action}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.fulfilled_order = 20
        self.dropped_order = 0
        self.rejected_order = 0
        self.profit = 0
        self.total_orders = 20
        self.available_drivers = self.starting_available_drivers

        self.pickup_durations = np.array([])
        self.driver_used_counter = self.start_driver_used_counter.copy()

        self.df_availability = self.availability_reset
        self.df_stats = self.stats_reset

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        results = {'Total orders': self.total_orders - 20,
                   'Avg. Early pickup time': self.avg_pickup_time,
                   'Fulfilled orders': self.fulfilled_order - 20,
                   'Dropped orders': self.dropped_order,
                   'Rejected orders': self.rejected_order,
                   'Profit': self.profit,
                   'Utilisation spread': self.spread}
        return results

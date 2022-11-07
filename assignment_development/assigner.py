"""
This script creates an Environment for a PPO reinforcement learning model to be trained for the
purposes of assigning taxi drivers to customer orders. Penalties and rewards are specified as well as
other environment conditions.
"""
import random
from typing import Tuple
import gym
from gym import spaces
import pandas as pd
import numpy as np
import numpy.typing as npt
from math import radians, cos, sin, asin, sqrt


class AssignmentEnv(gym.Env):
    """A Taxi Driver assignment environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, availability: str, stats: str, taxi_zones: str):
        """
        Initialise environment and read in datasets that define starting conditions
        :param availability: Path to starting driver availability
        :param stats: Path to starting driver stats
        :param taxi_zones: Path to taxi location coordinates
        """
        # Take copies of the imported datasets so that the environment can be reset
        self.availability_path = availability
        self.stats_path = stats
        self.taxi_path = taxi_zones

        self.df_availability = pd.read_csv(self.availability_path, index_col='Driver')
        self.df_stats = pd.read_csv(self.stats_path, index_col='Driver')

        self.taxi_zones = pd.read_csv(self.taxi_path)

        self.time_increment = 1  # New customer order occurs every 1 min

        # Environment start/stop conditions
        self.total_allowable_loss = -100
        self.max_profit = 1500

        # Current state of the environment

        # Drivers who are currently free
        self.available_drivers = self.df_availability[self.df_availability.Available == 'Y']["Available"].count()
        # Initialise variables
        self.order = None
        self.driver = None
        self.total_reward = 0
        self.fulfilled_order = 20
        self.avg_wait_time = 5
        self.dropped_order = 0
        self.rejected_order = 0
        self.profit = 0
        self.total_orders = 20
        self.reward = 0
        self.avg_pickup_time_after = 0
        self.late_pickup = 0

        # Environment Penalties
        # Penalise the model if drivers are being under utilised
        self.spread_before = self.df_stats.Utilisation.max() - self.df_stats.Utilisation.min()
        self.spread_after = self.df_stats.Utilisation.max() - self.df_stats.Utilisation.min()

        # Penalise the model if an order is rejected
        self.dropped_order_penalty = 100
        self.late_penalty = 25
        self.unfair_penalty = 25
        self.rejected_order_penalty = 15
        self.late_threshold = -15
        self.extreme_late_threshold = -45

        self.pickup_durations = np.array([])  # Array to store the collection wait times

        # Action space to choose the assignment algorithm
        # Most algorithms require a normalised action space
        self.action_space = spaces.Box(
            low=np.array([-1]), high=np.array([1]), shape=(1,), dtype=np.float16)

        # Observation space contains the business KPIs
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, 8), dtype=np.float16)

    def _haversine(self, PULocation: str, DOLocation: str, speed: float = 20.0) -> float:
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)

        :param PULocation: Starting Location ID
        :param DOLocation: Finishing location ID
        :param speed: Speed of travel in kph
        :return: Time to travel between locations in minutes
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
        return ((c * r) / speed) * 60  # Convert to minutes

    def _next_observation(self) -> npt.NDArray:
        """Generate a random customer order every timestep and store it as
        environment observations. These are normalised to be between 0-1."""

        self._customer_order()  # Generate customer order
        # Identify available drivers
        self.available_drivers = self.df_availability[self.df_availability.Available == 'Y']["Available"].count()

        # Create obs and normalise to between 0-1
        obs = np.array([
            self.fulfilled_order / self.total_orders,  # Fulfilled order %
            self.order.get("pickup_time") / 15,  # Max pickup time is 20 mins
            self.order.get("drop-off_time") / (30 + self.order.get("pickup_time")),
            self.dropped_order / self.total_orders,  # Dropped order %,
            self.rejected_order / self.total_orders,  # Dropped order %,
            self.df_stats.Utilisation.max() - self.df_stats.Utilisation.min(),  # Spread of driver utilisation
            self.available_drivers / len(self.df_availability),  # No. of available drivers
            self.late_pickup / self.total_orders  # No. of pickups longer than 15mins
        ])
        return obs

    def _customer_order(self):
        """Build a customer order"""
        while True:
            self.order = {"pickup_location": random.choice(self.taxi_zones.LocationID)}
            self.order.update({"drop-off_location": random.choice(self.taxi_zones.LocationID)})
            self.order.update(
                {"pickup_time": random.randint(5, 30)})  # Orders are ready to be picked up between 5 and 30 mins

            duration = self._haversine(self.order['pickup_location'], self.order['drop-off_location'])

            self.order.update({"drop-off_time": duration})  # Estimated Duration

            self.order.update({"fare": duration * 0.5})  # Earn 0.5 credit per minute of driving
            # Focus is on shorter trips
            if duration < 60:
                break

    def _update_driver_availability(self):
        """Update the driver availability lookup.
        This is to be run before every time step """
        self.df_availability.Time_until_free = self.df_availability.Time_until_free - self.time_increment
        self.df_availability.loc[(self.df_availability.Time_until_free <= 0),
                                 ['Time_until_free', "Available"]] = [0, 'Y']

    def _driver_assignment(self):
        """Once an order is assigned to a driver change their availability status.
        This is to be run after every time step."""

        self.df_availability.at[self.driver, 'Available'] = "N"
        # Sometimes a driver gets stuck in traffic and is out of commission for longer than expected
        if np.random.randint(0, 10) == 9:
            self.df_availability.at[self.driver, 'Time_until_free'] = self.order.get("drop-off_time") + 30
        else:
            self.df_availability.at[self.driver, 'Time_until_free'] = self.order.get("drop-off_time")

        # Increment their number of assigned deliveries
        self.df_stats.at[self.driver, 'Total_Fare'] = self.df_stats.at[self.driver, 'Total_Fare'] + self.order.get(
            "fare")

        # Set new location
        self.df_availability.loc[self.driver, 'Location'] = self.order.get('drop-off_location')

    def _random_driver_assignment(self):
        """This algorithm picks any available driver"""

        try:
            self.driver = self.df_availability[self.df_availability.Available == 'Y'].sample(1).index[0]
            driver_location = self.df_availability.loc[self.driver]['Location']

            duration = self._haversine(driver_location, self.order['drop-off_location'])

            # Extra reward given if order is picked up before estimated time
            time_delta = self.order.get("pickup_time") - duration

            if time_delta < self.late_threshold:
                self.late_pickup += 1
                return 0, time_delta
            else:
                self.fulfilled_order += 1
                return self.order.get("fare"), time_delta

        except ValueError:
            # If no driver is available then the order is treated as dropped
            self.dropped_order += 1
            return None, None

    def _closest_driver_assignment(self):
        """This algorithm picks the driver with the quickest lead time to pickup"""

        drivers = list(self.df_availability[self.df_availability.Available == 'Y'].index)
        if len(drivers) > 0:
            min_duration = 0
            for index, driver in enumerate(drivers):
                driver_location = self.df_availability.loc[driver]['Location']
                duration = self._haversine(driver_location, self.order['drop-off_location'])
                if index == 0:
                    min_duration = duration
                    self.driver = driver
                elif duration < min_duration:
                    min_duration = duration
                    self.driver = driver

            order_pickup_duration = min_duration
            time_delta = self.order.get("pickup_time") - order_pickup_duration

            if time_delta < self.late_threshold:
                self.late_pickup += 1
                return 0, time_delta
            else:
                self.fulfilled_order += 1
                return self.order.get("fare"), time_delta

        else:
            # If no driver is available then the order is treated as dropped
            self.dropped_order += 1
            return None, None

    def _lowest_utilisation_driver_assignment(self):
        """This algorithm picks the driver with the lowest earnings figure"""

        drivers = list(self.df_availability[self.df_availability.Available == 'Y'].index)
        if len(drivers) > 0:
            lowest_utilisation = self.df_stats[self.df_stats.index.isin(drivers)]["Utilisation"].min()
            self.driver = self.df_stats[self.df_stats["Utilisation"] == lowest_utilisation].index[0]
            driver_location = self.df_availability.loc[self.driver]['Location']
            duration = self._haversine(driver_location, self.order['drop-off_location'])

            time_delta = self.order.get("pickup_time") - duration

            if time_delta < self.late_threshold:
                self.late_pickup += 1
                return 0, time_delta
            else:
                self.fulfilled_order += 1
                return self.order.get("fare"), time_delta
        else:
            # If no driver is available then the order is treated as dropped
            self.dropped_order += 1
            return None, None

    def _refuse_order(self):
        """The model has the choice to reject the order before it is too late"""
        self.rejected_order += 1
        return None, True

    def _take_action(self, action: npt.NDArray) -> float:
        """
        Chose the assignment algorithm and update environment variables

        :param action: Assignment algorithm choice
        :return: Reward value
        """
        action_type = action[0]
        # Reset variables
        refuse = False
        reward = 0
        self.reward = 0
        order_pickup_duration = 0

        self.spread_before = self.df_stats.Utilisation.max() - self.df_stats.Utilisation.min()

        if action_type < -0.5:
            reward, refuse = self._refuse_order()

        elif -0.5 < action_type < 0:
            reward, order_pickup_duration = self._closest_driver_assignment()

        elif 0 < action_type < 0.5:
            reward, order_pickup_duration = self._lowest_utilisation_driver_assignment()

        elif action_type > 0.5:
            reward, order_pickup_duration = self._random_driver_assignment()

        # Update driver utilisation
        if reward is not None and refuse is False:
            self.df_stats['Utilisation'] = self.df_stats['Total_Fare'].apply(
                lambda x: x / max(self.df_stats.Total_Fare))

            # Calculate spread of utilisation figures
            self.spread_after = self.df_stats.Utilisation.max() - self.df_stats.Utilisation.min()

            # if order_pickup_duration is not None and refuse is False:
            if len(self.pickup_durations) > 0:
                self.avg_pickup_time_before = np.median(self.pickup_durations)

            else:
                self.avg_pickup_time_before = 0

            self.pickup_durations = np.append(self.pickup_durations, order_pickup_duration)
            self.avg_pickup_time_after = np.median(self.pickup_durations)

            spread_diff = self.spread_before > self.spread_after
            mean_time_diff = self.avg_pickup_time_before > self.avg_pickup_time_after

            self.reward += reward

            # Reduce reward if Utilisation range spread
            if spread_diff:
                self.reward -= self.unfair_penalty * 0.5
            else:
                self.reward += self.unfair_penalty * 0.5

            if order_pickup_duration < self.extreme_late_threshold:
                self.reward -= self.unfair_penalty

            if mean_time_diff:
                self.reward -= self.unfair_penalty * 0.25
            else:
                self.reward += self.unfair_penalty * 0.25

            self._driver_assignment()
            return self.reward
        elif refuse:
            return -self.rejected_order_penalty
        else:
            # Order is unfulfilled
            return -self.dropped_order_penalty

    def step(self, action: npt.NDArray) -> Tuple:
        self.total_reward = 0
        obs = self._next_observation()  # Update environment state

        self._update_driver_availability()  # Update current driver availability
        self.total_reward = self._take_action(action)  # Choose assignment algorithm

        self.profit += self.total_reward  # Calculate running profit
        self.total_orders += 1  # Increment total orders

        if self.profit <= self.total_allowable_loss:
            return obs, self.profit, True, {"Agent action": action}

        elif self.profit >= self.max_profit:
            return obs, self.profit, True, {"Agent action": action}
        else:
            return obs, self.total_reward, False, {"Agent action": action}

    def reset(self):
        """
        Reset the state of the environment to an initial state
        :return: Dict: New customer order
        """
        # Reset default values
        self.fulfilled_order = 20
        self.dropped_order = 0
        self.rejected_order = 0
        self.profit = 0
        self.total_orders = 20
        self.late_pickup = 0
        self.total_reward = 0

        # Create empty array for pickup times
        self.pickup_durations = np.array([])

        # Read in initial datasets
        # Using deepcopy on dataframes does no work
        self.df_availability = pd.read_csv(self.availability_path, index_col='Driver')
        self.df_stats = pd.read_csv(self.stats_path, index_col='Driver')

        # Recalculate the initialisation stats
        self.spread_before = self.df_stats.Utilisation.max() - self.df_stats.Utilisation.min()
        self.spread_after = self.df_stats.Utilisation.max() - self.df_stats.Utilisation.min()
        self.available_drivers = self.df_availability[self.df_availability.Available == 'Y']["Available"].count()

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment
        results = {'Total orders': self.total_orders - 20,
                   'Avg. Early pickup time': self.avg_pickup_time_after,
                   'Fulfilled orders': self.fulfilled_order - 20,
                   'Dropped orders': self.dropped_order,
                   'Rejected orders': self.rejected_order,
                   'Late orders': self.late_pickup,
                   'Profit': self.profit,
                   'Utilisation spread': self.spread_after}
        return results

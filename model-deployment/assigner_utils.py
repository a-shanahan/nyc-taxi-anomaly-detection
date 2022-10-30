import random
import numpy as np
import numpy.typing as npt
import mysql.connector as connector
from sqlalchemy import create_engine
import sys
from math import radians, cos, sin, asin, sqrt
import pandas as pd
from typing import Tuple, Dict


class AssignerUtils:
    def __init__(self, config: Dict, load=False):
        self.config = config
        self.late_threshold = 15
        self.fulfilled_order = 1
        self.dropped_order = 0
        self.refuse_order = 0
        self.late_pickup = 0
        self._db_connection()
        if load:
            self._load_data()
        self.order = {}
        self.all_drivers = self._query_execute("select count(Driver) from availability")[0][0]

    def _load_data(self):
        stats = pd.read_csv('../assignment_development/data/df_stats.csv', index_col='Driver')
        locations = pd.read_csv('../assignment_development/data/taxi_cords.csv', index_col='LocationID')
        availability = pd.read_csv('../assignment_development/data/driver_availability.csv', index_col='Driver')

        stats.to_sql('stats', con=self.engine, if_exists='replace')
        locations.to_sql('locations', con=self.engine, if_exists='replace')
        availability.to_sql('availability', con=self.engine, if_exists='replace')

        stats.to_sql('stats', con=self.engine, if_exists='replace')
        locations.to_sql('locations', con=self.engine, if_exists='replace')
        availability.to_sql('availability', con=self.engine, if_exists='replace')

        running_totals = pd.DataFrame({'fulfilled_order': [0],
                                       'dropped_order': [0],
                                       'rejected_order': [0],
                                       'late_pickup': [0],
                                       'refuse_order': [0]})
        running_totals.to_sql('running_totals', con=self.engine, if_exists='replace')

    def _db_connection(self):
        """
        Initialise connection to database
        """
        try:
            self.conn = connector.connect(host=self.config['host'],
                                          user=self.config['user'],
                                          password=self.config['password'],
                                          database=self.config['database'])
        except Exception as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)

        uri = f"mysql+mysqlconnector://{self.config.get('user')}:{self.config.get('password')}@" \
              f"{self.config.get('host')}/{self.config.get('database')}"

        self.engine = create_engine(uri)
        # Get Cursor
        self.cursor = self.conn.cursor()

    def _coordinate(self, coordinate: str, location: int) -> str:
        """
        Retrieve coordinate when given Location ID
        :param coordinate: Either Latitude or Longitude
        :param location: Location ID
        :return: Coordinate
        """
        query = "select " + coordinate + " from locations where LocationID = '" + str(location) + "'"
        # executing cursor
        self.cursor.execute(query)
        # display all records
        results = self.cursor.fetchall()
        return results[0][0]

    def _haversine(self, PULocation: int, DOLocation: int, speed: float = 20.0) -> float:
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)

        :param PULocation: Starting Location ID
        :param DOLocation: Finishing location ID
        :param speed: Speed of travel in kph
        :return: Time to travel between locations in minutes
        """

        lon1 = self._coordinate('Longitude', PULocation)
        lat1 = self._coordinate('Latitude', PULocation)

        lon2 = self._coordinate('Longitude', DOLocation)
        lat2 = self._coordinate('Latitude', DOLocation)

        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return ((c * r) / speed) * 60  # Convert to minutes

    def customer_order(self, new_order: Dict):
        """
        Build a customer order
        :param new_order: New customer order
        :return: Calls function to generate environment observations
        """
        self.order = {"pickup_location": new_order.get('PULocation')}
        self.order.update({"drop-off_location": new_order.get('DOLocation')})
        self.order.update(
            {"pickup_time": new_order.get('pickup_time')})  # Orders are ready to be picked up between 5 and 30 mins
        est_duration = self._haversine(self.order['pickup_location'], self.order['drop-off_location'])
        self.order.update({"drop-off_time": est_duration})  # Estimated Duration
        self.order.update({"fare": new_order.get('fare')})
        return self._next_observation()

    def _query_execute(self, query: str):
        """
        Execute query on database
        :param query: SQL query
        :return: Query results
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def _next_observation(self) -> npt.NDArray:
        """
        Generate a customer order every timestep and store it as
        environment observations. These are normalised to be between 0-1.
        :return: Normalised array containing environment observations
        """
        # Identify available drivers
        available_drivers = self._query_execute("select count(Driver) from availability where Available = 'Y'")[0][0]
        utilisation = self._query_execute("select MAX(Utilisation) - MIN(Utilisation) from stats")[0][0]

        total_orders = self.fulfilled_order + self.dropped_order + self.refuse_order + self.late_pickup
        # Create obs and normalise to between 0-1
        obs = np.array([
            self.fulfilled_order / total_orders,  # Fulfilled order %
            self.order.get("pickup_time") / 15,  # Max pickup time is 20 mins
            self.order.get("drop-off_time") / (30 + self.order.get("pickup_time")),
            self.dropped_order / total_orders,  # Dropped order %,
            self.refuse_order / total_orders,  # Dropped order %,
            utilisation,  # Spread of driver utilisation
            available_drivers / self.all_drivers,  # No. of available drivers
            self.late_pickup / total_orders  # No. of pickups longer than 15 mins
        ])
        return obs

    def driver_assignment(self, assignment: str, driver: str):
        """
        Once an order is assigned to a driver change their availability status.
        :param assignment: Y or N depending on assignment
        :param driver: Driver ID
        """
        self._query_execute("UPDATE availability SET Available = '" + assignment +
                            "' WHERE Driver = '" + driver + "'")

        # Drivers only become available when journey is completed
        if assignment == 'Y':
            self._query_execute("UPDATE stats SET Total_Fare = "
                                "Total_Fare + '" + self.order.get("fare") +
                                "' WHERE Driver = '" + driver + "'")
            self._query_execute("UPDATE availability SET Location = '" +
                                self.order.get('drop-off_location') +
                                "' WHERE Driver = '" + driver + "'")

    def random_driver_assignment(self) -> Tuple:
        """
        This algorithm picks any available driver
        :return: driver ID, time delta to order
        """
        try:
            result = self._query_execute("Select Driver, Location from availability where Available = 'Y'")
            driver, driver_location = random.choice(result)
            duration = self._haversine(driver_location, self.order['drop-off_location'])
            self.driver_assignment('Y', driver)
            # Extra reward given if order is picked up before estimated time
            time_delta = self.order.get("pickup_time") - duration
            if time_delta < self.late_threshold:
                self._query_execute("UPDATE running_totals SET late_pickup = late_pickup + 1")
                self.late_pickup = self._query_execute("Select late_pickup from running_totals")[0][0]
                return driver, time_delta
            else:
                self._query_execute("UPDATE running_totals SET fulfilled_order = fulfilled_order + 1")
                self.fulfilled_order = self._query_execute("Select fulfilled_order from running_totals")[0][0]
                return driver, time_delta

        except ValueError:
            # If no driver is available then the order is treated as dropped
            self._query_execute("UPDATE running_totals SET dropped_order = dropped_order + 1")
            self.dropped_order = self._query_execute("Select dropped_order from running_totals")[0][0]
            return None, None

    def closest_driver_assignment(self) -> Tuple:
        """
        This algorithm picks the driver with the quickest lead time to pickup
        :return: driver ID, time delta to order
        """
        results = self._query_execute("Select Driver, Location from availability where Available = 'Y'")
        if len(results) > 0:
            min_duration = None
            driver = None
            for free_driver, driver_location in results:
                duration = self._haversine(driver_location, self.order['drop-off_location'])
                if not min_duration:
                    min_duration = duration
                    driver = free_driver
                elif duration < min_duration:
                    min_duration = duration
                    driver = free_driver

            order_pickup_duration = min_duration
            time_delta = self.order.get("pickup_time") - order_pickup_duration
            self.driver_assignment('Y', driver)
            if time_delta < self.late_threshold:
                self._query_execute("UPDATE running_totals SET late_pickup = late_pickup + 1")
                self.late_pickup = self._query_execute("Select late_pickup from running_totals")[0][0]
                return driver, time_delta
            else:
                self._query_execute("UPDATE running_totals SET fulfilled_order = fulfilled_order + 1")
                self.fulfilled_order = self._query_execute("Select fulfilled_order from running_totals")[0][0]
                return driver, time_delta
        else:
            # If no driver is available then the order is treated as dropped
            self._query_execute("UPDATE running_totals SET dropped_order = dropped_order + 1")
            self.dropped_order = self._query_execute("Select dropped_order from running_totals")[0][0]
            return None, None

    def lowest_utilisation_driver_assignment(self) -> Tuple:
        """
        This algorithm picks the driver with the lowest earnings figure
        :return: driver ID, time delta to order
        """
        results = self._query_execute("Select a.Driver, Location "
                                      "from availability a "
                                      "LEFT JOIN "
                                      "stats s "
                                      "ON a.Driver=s.Driver "
                                      "where "
                                      "s.Utilisation in (Select MIN(Utilisation) from stats)")
        if len(results) > 0:
            driver, driver_location = results[0]
            duration = self._haversine(driver_location, self.order['drop-off_location'])
            time_delta = self.order.get("pickup_time") - duration
            self.driver_assignment('Y', driver)
            if time_delta < self.late_threshold:
                self._query_execute("UPDATE running_totals SET late_pickup = late_pickup + 1")
                self.late_pickup = self._query_execute("Select late_pickup from running_totals")[0][0]
                return driver, time_delta
            else:
                self._query_execute("UPDATE running_totals SET fulfilled_order = fulfilled_order + 1")
                self.fulfilled_order = self._query_execute("Select fulfilled_order from running_totals")[0][0]
                return driver, time_delta
        else:
            # If no driver is available then the order is treated as dropped
            self._query_execute("UPDATE running_totals SET dropped_order = dropped_order + 1")
            self.dropped_order = self._query_execute("Select dropped_order from running_totals")[0][0]
            return None, None

    def refuse(self):
        """
        The model has the choice to reject the order
        """
        self._query_execute("UPDATE running_totals SET refuse_order = refuse_order + 1")
        self.refuse_order = self._query_execute("Select refuse_order from running_totals")[0][0]
        return None, True

import numpy as np
import numpy.typing as npt
import mysql.connector as connector
from sqlalchemy import create_engine
import sys
from math import radians, cos, sin, asin, sqrt


class AssignerUtils:
    def __init__(self):
        self._db_connection()
        self.order = {}
        self.all_drivers = self._query_execute("select count(Driver) from availability")[0]

    def _db_connection(self):
        try:
            self.conn = connector.connect(
                host="localhost",
                user='newuser',
                password='newpassword',
                database="demo"
            )
        except Exception as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)

        uri = 'mysql+mysqlconnector://newuser:newpassword@localhost/demo'

        self.engine = create_engine(uri)
        # Get Cursor
        self.cursor = self.conn.cursor()

    def _coordinate(self, coordinate, location):
        query = "select " + coordinate + " from locations where LocationID = " + Location
        # executing cursor
        self.cursor.execute(query)
        # display all records
        results = self.cursor.fetchall()
        return results[0]

    def _haversine(self, PULocation: str, DOLocation: str, speed: float = 20.0) -> float:
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

    def customer_order(self, PULocation, DOLocation, pickup_time, fare):
        """Build a customer order"""
        self.order = {"pickup_location": PULocation}
        self.order.update({"drop-off_location": DOLocation})
        self.order.update(
            {"pickup_time": pickup_time})  # Orders are ready to be picked up between 5 and 30 mins
        est_duration = self._haversine(self.order['pickup_location'], self.order['drop-off_location'])
        self.order.update({"drop-off_time": est_duration})  # Estimated Duration
        self.order.update({"fare": fare})

    def _query_execute(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def _update_stats(self):
        self.fulfilled_order = 0
        self.total_orders = 0
        self.dropped_order = 0
        self.rejected_order = 0
        self.late_pickup = 0

    def next_observation(self) -> npt.NDArray:
        """Generate a customer order every timestep and store it as
        environment observations. These are normalised to be between 0-1."""
        self.customer_order()  # Generate customer order
        # Identify available drivers
        available_drivers = self._query_execute("select count(Driver) from availability where Available = Y")[0]
        utilisation = self._query_execute("select MAX(Utilisation) - MIN(Utilisation) from drivers")[0]

        # Create obs and normalise to between 0-1
        obs = np.array([
            self.fulfilled_order / self.total_orders,  # Fulfilled order %
            self.order.get("pickup_time") / 15,  # Max pickup time is 20 mins
            self.order.get("drop-off_time") / (30 + self.order.get("pickup_time")),
            self.dropped_order / self.total_orders,  # Dropped order %,
            self.rejected_order / self.total_orders,  # Dropped order %,
            utilisation,  # Spread of driver utilisation
            available_drivers / self.all_drivers,  # No. of available drivers
            self.late_pickup / self.total_orders  # No. of pickups longer than 15 mins
        ])
        return obs

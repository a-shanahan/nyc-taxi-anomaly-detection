import mysql.connector as connector
from sqlalchemy import create_engine
import sys

import pandas as pd

# Connect to MariaDB Platform
try:
    conn = connector.connect(
        host="localhost",
        user='newuser',
        password='newpassword',
        database="demo"
    )
    print(f"Connected to MariaDB Platform")
except Exception as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

uri = 'mysql+mysqlconnector://newuser:newpassword@localhost/demo'

engine = create_engine(uri)
# Get Cursor
cursor = conn.cursor()
print(cursor)

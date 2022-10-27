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
except Exception as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

uri = 'mysql+mysqlconnector://newuser:newpassword@localhost/demo'

engine = create_engine(uri)
# Get Cursor
cursor = conn.cursor()

df = pd.read_csv('../assignment_development/data/df_stats.csv', index_col=False)
locations = pd.read_csv('../assignment_development/data/taxi_cords.csv', index_col=False)

df.to_sql('driver', con=engine, if_exists='replace')
locations.to_sql('locations', con=engine, if_exists='replace')

query1 = "select * from driver where Driver ='db2d7f2629ee45a4849d0ea9ac143765'"

# executing cursor
cursor.execute(query1)

# display all records
table = cursor.fetchall()

# describe table
print('\n Query Results:')
for attr in table:
    print(attr)

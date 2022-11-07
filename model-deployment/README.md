# NYC Taxi System Deployment

Directory contents:

- Data: MariaDB config file
- Spark-cluster: Spark docker file and start up script
- Stream-setup: PySpark script to be executed on Spark cluster
- Utilities: Helper functions including assigner model, db connection and kafka monitoring.

The following steps will set up the network and then start the data streaming and processing.

## Pre-requisites
- Docker
- Docker-compose

## Docker compose

The [docker compose](model-deployment/docker-compose.yml) file will create the following containers:

| Container    | Exposed ports  |
|--------------|----------------|
| spark-master | 9090, 7077     |
| spark-worker | 9091, 7002     |
| kafka        | 29002, 9092    |
| zookeeper    | 22181          |
| mariadb      | 3306           |

To run the network in detached mode use the ```-d``` flag at the end of the command:  
```shell
docker-compose up
```

## Data streaming and processing
To replicate a system where new orders are raised and completed journeys are submitted a number of Python 
scripts have been developed. First, create a virtual environment and install the required libraries:

```shell
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Each script can then be started using:
```shell
python3 <script name>/.py
```

A summary of each script is provided at the top of each file but a high level overview is provided below:
- Anomaly detection consumer: Monitors completed journeys for anomalies
- Assign driver: Assigns drivers to customer orders
- Driver status: Updates database to reflect current driver availability
- Hold driver: Generates completed journeys at calculated intervals
- Order generation: Reads data from a NYC taxi parquet file to replicate new customer orders and journey details

Scripts can be started in any order but it is recommended that the order_generation.py file is started last. 

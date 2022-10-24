#!/bin/bash
/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 --driver-memory 1G --executor-memory 1G process_stream.py
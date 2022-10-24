from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

appName = "NYC-Taxis"
CHECKPOINT_LOCATION = "/tmp"

spark = SparkSession.builder. \
    appName(appName) \
    .config('spark.jars.packages',
            ['org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0']) \
    .config('spark.streaming.stopGracefullyOnShutdown', 'true') \
    .config("spark.driver.memory", "8G") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
kafka_servers = "kafka:9092"


df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_servers) \
    .option("subscribe", "completed-journey") \
    .load()

sample_schema = (
    StructType()
    .add("uid", StringType())
    .add("VendorID", IntegerType())
    .add("tpep_pickup_datetime", TimestampType())
    .add("tpep_dropoff_datetime", TimestampType())
    .add("passenger_count", DoubleType())
    .add("trip_distance", DoubleType())
    .add("RatecodeID", DoubleType())
    .add("store_and_fwd_flag", StringType())
    .add("PULocationID", IntegerType())
    .add("DOLocationID", IntegerType())
    .add("payment_type", IntegerType())
    .add("fare_amount", DoubleType())
    .add("extra", DoubleType())
    .add("mta_tax", DoubleType())
    .add("tip_amount", DoubleType())
    .add("tolls_amount", DoubleType())
    .add("total_amount", DoubleType())
    .add("improvement_surcharge", DoubleType())
    .add("congestion_surcharge", DoubleType())
    .add("airport_fee", IntegerType())
)

df = df.select(
    from_json(col("value").cast("string"), sample_schema).alias("value"),
).select("value.*")

keep = ['uid',
        'tpep_dropoff_datetime', 'tpep_pickup_datetime',
        'PULocationID', 'DOLocationID',
        'passenger_count',
        'trip_distance', 'payment_type', 'fare_amount',
        'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'improvement_surcharge']

# Drop columns that we are not interested in
parDF = df.select(*keep)

# Calculate journey time
parDF = parDF.withColumn('tpep_dropoff_datetime', parDF['tpep_dropoff_datetime'].cast("timestamp"))
parDF = parDF.withColumn('tpep_pickup_datetime', parDF['tpep_pickup_datetime'].cast("timestamp"))

# Calculate journey time
parDF = parDF.withColumn('trip_duration',
                         (parDF['tpep_dropoff_datetime'].cast("long") - parDF['tpep_pickup_datetime'].
                          cast("long")) / 60)


# Extract hour and weekday from timestamp
parDF = parDF.withColumn('pickup_hour', hour('tpep_pickup_datetime'))
parDF = parDF.withColumn('pickup_weekday', dayofweek('tpep_pickup_datetime'))

# Don't need start/finish timestamp so drop
drop_cols = ['tpep_dropoff_datetime', 'tpep_pickup_datetime']
parDF = parDF.drop(*drop_cols)

# Create new feature combining start and end location
parDF = parDF.withColumn('journey', concat(col('PULocationID').cast('string'), col('DOLocationID').cast('string')))

# Read in journey counts lookup and join onto data
journeyDF = spark.read.parquet('/data/journey_counts.parquet/part-00000-89624733-7737-4559-a0c9-6d15bea38f05-c000.snappy.parquet')

# Convert Journey ID to numeric value
parDF = parDF.join(journeyDF, on=['journey'], how='left')
parDF = parDF.drop(*['count', 'journey', 'PULocationID', 'DOLocationID'])
parDF = parDF.na.fill(value=0, subset=["passenger_count"])

to_scale = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
            'trip_duration']
one_hot = ['payment_type', 'passenger_count', 'pickup_hour', 'pickup_weekday']

scale_model = PipelineModel.load("/data/pipelines/scale_pipeline")

parDF = scale_model.transform(parDF)

parDF = parDF.drop(*[col + '_vec' for col in to_scale])

one_hot_model = PipelineModel.load("/data/pipelines/one_hot_pipeline")

parDF = one_hot_model.transform(parDF)

# Drop unwanted columns
drop_cols_1 = [col + '_indexed' for col in one_hot]
drop_cols_2 = [col + '_indexed_encoded' for col in one_hot]

drop_cols_3 = drop_cols_1 + drop_cols_2 + to_scale + one_hot
parDF = parDF.drop(*drop_cols_3)

# Extract scaled values from scaled columns
firstelement = udf(lambda v: float(v[0]), FloatType())

for c in parDF.columns:
    if c != 'journeyFreq' and c != 'uid' and c != 'features':
        parDF = parDF.withColumn(c, firstelement(c))

# One Hot encoding produces sparse vector, convert to array/
# This makes it easier to load into TensorFlow later
parDF = parDF.withColumn("features", vector_to_array('features'))

parDF = parDF.select(*parDF.columns, *[parDF.features[i].cast(FloatType()) for i in range(56)])
parDF = parDF.drop('features')

cols = parDF.columns
df = parDF.withColumn("value", to_json(struct([x for x in cols]))).drop(*cols)

df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_servers) \
    .option("checkpointLocation", CHECKPOINT_LOCATION) \
    .option("topic", "journey-finished") \
    .start() \
    .awaitTermination()

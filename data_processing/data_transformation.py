import glob
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.functions import vector_to_array

files = [x for x in glob.glob('/data/raw_data/*.parquet')]

# Shuffle files
random.shuffle(files)

with open('/data/processed_files.txt', 'w') as f:
    for line in files[0:6]:
        f.write(f"{line}\n")

spark = SparkSession.builder \
    .appName("NYC") \
    .getOrCreate()

parDF = spark.read.parquet(*files[0:6])

keep = ['tpep_dropoff_datetime', 'tpep_pickup_datetime',
        'PULocationID', 'DOLocationID',
        'passenger_count',
        'trip_distance', 'payment_type', 'fare_amount',
        'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'improvement_surcharge']

# Drop columns that we are not interested in
parDF = parDF.select(*keep)

# Calculate journey time
parDF = parDF.withColumn('trip_duration',
                         (parDF['tpep_dropoff_datetime'].cast("long") - parDF['tpep_pickup_datetime'].
                          cast("long")) / 60)

# Drop unusually long journeys in both time and distance. Determined during EDA
parDF = parDF.filter(parDF['trip_distance'] < 3).filter(parDF['trip_duration'].between(0, 20))

# Extract hour and weekday from timestamp
parDF = parDF.withColumn('pickup_hour', hour('tpep_pickup_datetime'))
parDF = parDF.withColumn('pickup_weekday', dayofweek('tpep_pickup_datetime'))
# Parition data by month
parDF = parDF.withColumn('pickup_month', month('tpep_pickup_datetime'))

# Don't need start/finish timestamp so drop
drop_cols = ['tpep_dropoff_datetime', 'tpep_pickup_datetime']
parDF = parDF.drop(*drop_cols)

# Create new feature combining start and end location
parDF = parDF.withColumn('journey', concat(col('PULocationID').cast('string'), col('DOLocationID').cast('string')))

# Large number of categories in journey so use frequency probability as proxy instead on OneHotEncoding
journey_df = parDF.groupBy('journey').count()

# This needs to be saved as a lookup for use during inference
journey_df = journey_df.withColumn('journeyFreq', format_number(journey_df['count'] / parDF.count(), 8))
journey_df.write.parquet('/data/journey_counts.parquet')


# Convert Journey ID to numeric value
parDF = parDF.join(journey_df, on=['journey'], how='left')
parDF = parDF.drop(*['count', 'journey', 'PULocationID', 'DOLocationID'])
parDF = parDF.na.fill(value=0, subset=["passenger_count"])

# Scale continuous columns
# One hot categories


# Not going to keep pickup month. Training data won't cover all months
to_scale = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
            'trip_duration']
one_hot = ['payment_type', 'passenger_count', 'pickup_hour', 'pickup_weekday']


# Scaling pipeline

assembler = [VectorAssembler(inputCols=[col], outputCol=col + '_vec') for col in to_scale]
scale = [MinMaxScaler(inputCol=col + '_vec', outputCol=col + '_scaled') for col in to_scale]
pipe = Pipeline(stages=assembler + scale)

scale_model = pipe.fit(parDF)
scale_model.save("/data/pipelines/scale_pipeline")

parDF = scale_model.transform(parDF)

# Drop unwanted columns
parDF = parDF.drop(*[col + '_vec' for col in to_scale])


# One hot encoded pipeline

indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in one_hot]

encoders = [OneHotEncoder(dropLast=False, inputCol=indexer.getOutputCol(),
                          outputCol="{0}_encoded".format(indexer.getOutputCol()))
            for indexer in indexers]

# Vectorizing encoded values
assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol="features")
pipeline = Pipeline(stages=indexers + encoders + [assembler])

one_hot_model = pipeline.fit(parDF)
one_hot_model.save("/data/pipelines/one_hot_pipeline")

parDF = one_hot_model.transform(parDF)

row = parDF.select("features").head()
no_features = [i.size for i in row(0).asDict()][0]

# Drop unwanted columns
drop_cols_1 = [col + '_indexed' for col in one_hot]
drop_cols_2 = [col + '_indexed_encoded' for col in one_hot]

drop_cols_3 = drop_cols_1 + drop_cols_2 + to_scale + one_hot
parDF = parDF.drop(*drop_cols_3)

# Extract scaled values from scaled columns
firstelement = udf(lambda v: float(v[0]), FloatType())

for c in parDF.columns:
    if c != 'journeyFreq' and c != 'features' and c != 'pickup_month' and c != 'features':
        parDF = parDF.withColumn(c, firstelement(c))

# One Hot encoding produces sparse vector, convert to array/
# This makes it easier to load into TensorFlow later
parDF = parDF.withColumn("features", vector_to_array('features'))

# Extract values from array and store as columns
parDF = parDF.select(*parDF.columns, *[parDF.features[i].cast(FloatType()) for i in range(no_features)])
parDF = parDF.drop('features')

# Write data and partition by weekday
parDF.write.partitionBy("pickup_month").parquet('/data/parDF.parquet')

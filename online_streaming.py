from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_json, struct
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("DiabetesStreaming") \
    .getOrCreate()

model = PipelineModel.load("best_model")

schema = StructType([
    StructField("HighBP", DoubleType()),
    StructField("HighChol", DoubleType()),
    StructField("CholCheck", DoubleType()),
    StructField("BMI", DoubleType()),
    StructField("Smoker", DoubleType()),
    StructField("Stroke", DoubleType()),
    StructField("HeartDiseaseorAttack", DoubleType()),
    StructField("PhysActivity", DoubleType()),
    StructField("Fruits", DoubleType()),
    StructField("Veggies", DoubleType()),
    StructField("HvyAlcoholConsump", DoubleType()),
    StructField("AnyHealthcare", DoubleType()),
    StructField("NoDocbcCost", DoubleType()),
    StructField("GenHlth", DoubleType()),
    StructField("MentHlth", DoubleType()),
    StructField("PhysHlth", DoubleType()),
    StructField("DiffWalk", DoubleType()),
    StructField("Sex", DoubleType()),
    StructField("Age", DoubleType()),
    StructField("Education", DoubleType()),
    StructField("Income", DoubleType())
])

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "health_data") \
    .option("startingOffsets", "latest") \
    .load()

parsed_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

predictions = model.transform(parsed_df)

output = predictions.select("*", col("prediction").alias("predicted_diabetes"))

kafka_output = output.select(
    to_json(struct("*")).alias("value")
)

query = kafka_output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "health_data_predicted") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .start()

query.awaitTermination()
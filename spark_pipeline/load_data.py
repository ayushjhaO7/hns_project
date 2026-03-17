from pyspark.sql import SparkSession
import os

def create_spark_session(app_name):
    os.environ['HADOOP_HOME'] = "C:/hadoop"
    os.environ['hadoop.home.dir'] = "C:/hadoop"
    
    # Use a simpler URI first to avoid Windows shell parsing errors
    # Ensure you replace <username>, <password>, and <cluster> with your Atlas info
    
    mongo_uri = "mongodb+srv://ayushjha:ayushjha007@cluster0.aoh7j5n.mongodb.net/"

    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.mongodb.output.uri", mongo_uri) \
        .config("spark.mongodb.input.uri", mongo_uri) \
        .getOrCreate()

def load_crime_dataset(spark, file_path):
    # Read CSV, utilizing Spark's schema inference
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

def load_distributed_data(spark):
    """
    Fetches the combined results from both laptops stored in MongoDB Atlas.
    """
    print("Fetching data from Distributed MongoDB Nodes...")
    
    # Spark format for MongoDB connector
    df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    
    # Distributed cleaning: Remove duplicates from multi-node ingestion
    df = df.dropDuplicates(["STATE/UT", "DISTRICT"])
    
    print(f"Total Distributed Records Loaded from Cloud: {df.count()}")
    return df
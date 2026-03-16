from pyspark.sql import SparkSession

def load_crime_data():

    spark = SparkSession.builder \
        .appName("CrimeHotspotAnalysis") \
        .getOrCreate()

    df = spark.read.csv(
        "D:/Project/Crime-hotspot/data/crime_dataset.csv",
        header=True,
        inferSchema=True
    )

    return spark, df
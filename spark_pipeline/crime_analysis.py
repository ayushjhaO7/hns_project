from pyspark.sql import SparkSession
from pyspark.sql.functions import sum

spark = SparkSession.builder.appName("CrimeAnalysis").getOrCreate()

df = spark.read.csv(
    "D:/Project/Crime-hotspot/data/crime_dataset.csv",
    header=True,
    inferSchema=True
)

# Total crimes per state
state_crimes = df.groupBy("STATE/UT").agg(sum("crime_count").alias("Total_Crimes"))

state_crimes.show()

top_states = state_crimes.orderBy("Total_Crimes", ascending=False)

top_states.show(10)

top_states = state_crimes.orderBy("Total_Crimes", ascending=False)

top_states.show(10)

crime_type = df.groupBy("crime_type").sum("crime_count")

crime_type.show()



spark.stop()
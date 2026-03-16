from pyspark.sql.functions import sum

def create_features(df):

    district_crime = df.groupBy("STATE/UT","DISTRICT") \
                       .agg(sum("crime_count").alias("total_crime"))

    return district_crime
from pyspark.sql.functions import to_timestamp, month, year, dayofweek, hour, col

def engineer_temporal_features(df, date_col="date"):
    # Extract temporal trends if a timestamp column exists
    if date_col in df.columns:
        # Cast to proper timestamp if it's currently a string
        df = df.withColumn("timestamp", to_timestamp(col(date_col)))
        
        # Create granular time features for analysis
        df = df.withColumn("year", year("timestamp"))
        df = df.withColumn("month", month("timestamp"))
        df = df.withColumn("day_of_week", dayofweek("timestamp"))
        df = df.withColumn("hour", hour("timestamp"))
        
    return df
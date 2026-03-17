from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql.functions import col, month, year, to_date

def load_and_prepare_data(spark, file_path):
    # Load distributed data
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Feature Engineering (Example: extracting month/year if Date exists)
    if "Date" in df.columns:
        df = df.withColumn("Date", to_date(col("Date")))
        df = df.withColumn("Month", month(col("Date")))
        df = df.withColumn("Year", year(col("Date")))

    # Categorical Indexing (e.g., District, Crime_Type)
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
        for c in ["District", "Crime_Type"] if c in df.columns
    ]
    
    for indexer in indexers:
        df = indexer.fit(df).transform(df)

    # Assemble numerical features into a single vector
    feature_cols = ["Latitude", "Longitude"] + [f"{c}_index" for c in ["District", "Crime_Type"] if c in df.columns]
    
    # Add time features if they were created
    if "Month" in df.columns: feature_cols.extend(["Month", "Year"])

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
    df = assembler.transform(df)

    # Scale the features
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    df = scaler.fit(df).transform(df)

    # Return required columns
    return df.select("features", col("Is_Hotspot").alias("label"), "Latitude", "Longitude", "District")
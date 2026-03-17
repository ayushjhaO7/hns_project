from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

def detect_hotspots(df, k_clusters=5):
    # Cluster based purely on the total crime volume per district
    assembler = VectorAssembler(
        inputCols=["total_crime"],
        outputCol="features_raw",
        handleInvalid="skip"
    )

    data = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True, 
        withMean=False
    )

    scaled_model = scaler.fit(data)
    scaled_data = scaled_model.transform(data)

    kmeans = KMeans(k=k_clusters, seed=42, featuresCol="features", predictionCol="hotspot_cluster")
    
    model = kmeans.fit(scaled_data)
    clusters = model.transform(scaled_data)

    return clusters, model
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

def detect_hotspots(df):

    assembler = VectorAssembler(
        inputCols=["total_crime"],
        outputCol="features_raw"
    )

    data = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features"
    )

    scaled_data = scaler.fit(data).transform(data)

    kmeans = KMeans(k=5, seed=1)

    model = kmeans.fit(scaled_data)

    clusters = model.transform(scaled_data)

    return clusters
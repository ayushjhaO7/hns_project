from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

def detect_hotspots(df):

    assembler = VectorAssembler(
        inputCols=["total_crime"],
        outputCol="features"
    )

    feature_data = assembler.transform(df)

    kmeans = KMeans(k=5, seed=1)

    model = kmeans.fit(feature_data)

    clusters = model.transform(feature_data)

    return clusters
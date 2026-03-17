from pyspark.ml.classification import GBTClassifier

def train_gbt(train_data):
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20)
    return gbt.fit(train_data)
from pyspark.ml.classification import RandomForestClassifier

def train_rf(train_data):
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=5)
    return rf.fit(train_data)
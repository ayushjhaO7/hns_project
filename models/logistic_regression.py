from pyspark.ml.classification import LogisticRegression

def train_lr(train_data):
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    return lr.fit(train_data)
from pyspark.ml.classification import LogisticRegression

def logistic_model(data):

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="total_crime"
    )

    model = lr.fit(data)

    predictions = model.transform(data)

    return predictions
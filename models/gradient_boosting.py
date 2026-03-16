from pyspark.ml.regression import GBTRegressor

def gradient_boosting_model(data):

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="total_crime"
    )

    model = gbt.fit(data)

    predictions = model.transform(data)

    return predictions
from pyspark.ml.regression import RandomForestRegressor

def random_forest_model(data):

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="total_crime"
    )

    model = rf.fit(data)

    predictions = model.transform(data)

    return predictions
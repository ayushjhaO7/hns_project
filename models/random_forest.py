from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler

def train_random_forest(df):

    assembler = VectorAssembler(
        inputCols=["YEAR"],
        outputCol="features"
    )

    data = assembler.transform(df)

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="crime_count"
    )

    model = rf.fit(data)

    predictions = model.transform(data)

    return predictions
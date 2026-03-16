from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler

def train_gbt(df):

    assembler = VectorAssembler(
        inputCols=["YEAR"],
        outputCol="features"
    )

    data = assembler.transform(df)

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="crime_count"
    )

    model = gbt.fit(data)

    predictions = model.transform(data)

    return predictions
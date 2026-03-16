from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

def train_logistic(df):

    assembler = VectorAssembler(
        inputCols=["YEAR"],
        outputCol="features"
    )

    data = assembler.transform(df)

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="crime_count"
    )

    model = lr.fit(data)

    predictions = model.transform(data)

    return predictions
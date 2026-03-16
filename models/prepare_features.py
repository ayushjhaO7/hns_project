from pyspark.ml.feature import VectorAssembler

def prepare_ml_features(df):

    assembler = VectorAssembler(
        inputCols=["YEAR"],
        outputCol="features"
    )

    data = assembler.transform(df)

    return data
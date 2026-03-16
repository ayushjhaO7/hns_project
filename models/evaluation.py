from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_model(predictions):

    evaluator = RegressionEvaluator(
        labelCol="total_crime",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    print("RMSE:", rmse)
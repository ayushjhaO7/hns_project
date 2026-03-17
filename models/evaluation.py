from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

def evaluate_model(predictions, model_name="Model"):
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"--- {model_name} Evaluation ---")
    print(f"Area Under ROC (AUC): {auc:.4f}\n")
    return auc

from pyspark.ml.evaluation import ClusteringEvaluator

def evaluate_kmeans(predictions_df):
    # The ClusteringEvaluator uses the Silhouette Score by default
    evaluator = ClusteringEvaluator(
        predictionCol="hotspot_cluster",
        featuresCol="features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    score = evaluator.evaluate(predictions_df)
    
    print("\n" + "="*30)
    print("      MODEL EVALUATION")
    print("="*30)
    print(f"K-Means Silhouette Score: {score:.4f}")
    print("(Score ranges from -1 to 1. Closer to 1 is better!)")
    print("="*30 + "\n")
    
    return score

def evaluate_supervised_model(predictions_df, model_name="Supervised Model"):
    # Note: Spark ML expects the true answer column to be named "label" 
    # and the model's guess to be named "prediction"
    
    # 1. Calculate Accuracy and F1-Score
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = multi_evaluator.evaluate(predictions_df)
    
    multi_evaluator.setMetricName("f1")
    f1_score = multi_evaluator.evaluate(predictions_df)

    print("\n" + "="*35)
    print(f"   {model_name} EVALUATION")
    print("="*35)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1_score:.4f} (Better for imbalanced data)")

    # 2. Calculate ROC-AUC (Only applicable if it's a Yes/No binary prediction)
    # 
    try:
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc = binary_evaluator.evaluate(predictions_df)
        print(f"ROC-AUC:  {auc:.4f}")
    except Exception:
        # Fails silently if the model is predicting 5 classes instead of 2
        pass 

    print("="*35 + "\n")
    return accuracy, f1_score
import os
from pyspark.sql.functions import col
from spark_pipeline.load_data import create_spark_session, load_crime_dataset
from spark_pipeline.preprocessing import clean_data
from spark_pipeline.crime_analysis import prepare_clustering_data, export_results
from models.hotspot_prediction import detect_hotspots
from models.evaluation import evaluate_kmeans, evaluate_supervised_model

from spark_pipeline.load_data import create_spark_session, load_distributed_data

# ML Imports
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier, OneVsRest
from pyspark.ml.feature import VectorAssembler

def main():
    spark = create_spark_session("DistributedAnalysis")

    # Local CSV ki jagah Cloud se data uthane ke liye:
    combined_df = load_distributed_data(spark)
    
    # Ab combined_df par aap analysis ya machine learning run kar sakte hain
    combined_df.show()

    # 2. PREPROCESSING (Using RDD Transformation inside clean_data)
    print("Pre-processing data using RDD map transformations...")
    clean_df = clean_data(raw_df)

    # 3. CRIME ANALYSIS (Using Spark SQL inside prepare_clustering_data)
    print("Running District Analysis using Spark SQL engine...")
    clustering_df = prepare_clustering_data(clean_df)

    # 4. UNSUPERVISED LEARNING: K-Means Hotspot Detection
    print("Running K-Means algorithm for Hotspot Discovery...")
    hotspot_predictions, kmeans_model = detect_hotspots(clustering_df, k_clusters=5)
    
    print("Evaluating K-Means performance...")
    evaluate_kmeans(hotspot_predictions)

    # 5. SUPERVISED LEARNING PHASE
    print("Preparing data for Supervised Models...")
    
    # Select columns including geographic labels for the final map
    ml_input = hotspot_predictions.select(
        "STATE/UT", 
        "DISTRICT", 
        "total_crime", 
        col("hotspot_cluster").alias("label")
    )
    
    # Standardize data for ML
    assembler = VectorAssembler(inputCols=["total_crime"], outputCol="features")
    ml_data = assembler.transform(ml_input)
    train_df, test_df = ml_data.randomSplit([0.8, 0.2], seed=42)

    # Model A: Random Forest
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20)
    rf_model = rf.fit(train_df)
    rf_predictions = rf_model.transform(test_df)
    evaluate_supervised_model(rf_predictions, "Random Forest")

    # Model B: Logistic Regression (Winning Model)
    print("Training Logistic Regression Model...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    lr_model = lr.fit(train_df)
    lr_predictions = lr_model.transform(test_df)
    evaluate_supervised_model(lr_predictions, "Logistic Regression")
    
    # Model C: GBT Classifier
    
    # print("Training Gradient Boosted Trees (GBT)...")
    # gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)
    # ovr = OneVsRest(classifier=gbt, featuresCol="features", labelCol="label")
    # ovr_model = ovr.fit(train_df)
    # gbt_predictions = ovr_model.transform(test_df)
    # evaluate_supervised_model(gbt_predictions, "Gradient Boosting (GBT)")
    
    

    # 6. FINAL EXPORT TO MONGODB ATLAS
    print("Exporting results to MongoDB Atlas for cloud storage...")
    
    # Prepare the final output set (Select only clean columns, no vectors)
    final_output = lr_predictions.select(
        "STATE/UT", 
        "DISTRICT", 
        "total_crime", 
        col("label").alias("actual_cluster"),
        col("prediction").alias("predicted_risk_level")
    )

    # Write to MongoDB Atlas
    # Note: 'overwrite' mode will replace the collection every time you run the pipeline
    final_output.write.format("com.mongodb.spark.sql.DefaultSource") \
        .mode("append") \
        .option("database", "crime_db") \
        .option("collection", "hotspots") \
        .save()

    print(f"✅ Pipeline finished! AI results are now live in MongoDB Atlas collection: crime_db.hotspots")
    spark.stop()

if __name__ == "__main__":
    main()
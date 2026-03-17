from pyspark.sql.functions import sum, desc
import pandas as pd

def prepare_clustering_data(df):
    """
    Consolidates crime counts per district using the Spark SQL engine.
    """
    # Register the DataFrame as a Temporary View for SQL queries
    df.createOrReplaceTempView("crime_data")

    # SQL Query: Aggregating total crime per district
    sql_query = """
        SELECT 
            `STATE/UT`, 
            DISTRICT, 
            SUM(crime_count) as total_crime 
        FROM 
            crime_data 
        GROUP BY 
            `STATE/UT`, DISTRICT
    """
    
    clustering_df = df.sparkSession.sql(sql_query)
    
    # Run a quick side-analysis for the log output
    print("\n--- Top 5 Districts by Crime Volume (Spark SQL) ---")
    df.sparkSession.sql("SELECT DISTRICT, SUM(crime_count) as total FROM crime_data GROUP BY DISTRICT ORDER BY total DESC LIMIT 5").show()
    
    return clustering_df

def aggregate_for_dashboard(df):
    # Create a high-level summary for R or Power BI
    if "DISTRICT" in df.columns and "YEAR" in df.columns:
        agg_df = df.groupBy("DISTRICT", "YEAR", "crime_type") \
                   .agg(sum("crime_count").alias("incident_count")) \
                   .orderBy(desc("incident_count"))
        return agg_df
    return df

def export_results(df, output_path, format="csv"):
    """
    Cleans ML columns, maps clusters to risk levels (0-4), and saves via Pandas.
    """
    # 1. Drop complex ML vectors to make the CSV readable
    columns_to_drop = ["features_raw", "features", "rawPrediction", "probability"]
    clean_export_df = df.drop(*[c for c in columns_to_drop if c in df.columns])
    
    # 2. Convert to Pandas for local saving (Windows workaround)
    pandas_df = clean_export_df.toPandas()
    
    # 3. Identify the cluster column
    cluster_col = "actual_cluster" if "actual_cluster" in pandas_df.columns else "hotspot_cluster"
    
    # 4. Sorting logic: Ensure Cluster 0 is the lowest crime and 4 is highest
    if cluster_col in pandas_df.columns and "total_crime" in pandas_df.columns:
        cluster_averages = pandas_df.groupby(cluster_col)['total_crime'].mean().sort_values()
        # Map: lowest average crime cluster -> 0, highest average -> 4
        risk_mapping = {old_id: new_risk_level for new_risk_level, old_id in enumerate(cluster_averages.index)}
        pandas_df['predicted_risk_level'] = pandas_df[cluster_col].map(risk_mapping)
    
    # 5. Save final CSV
    final_path = f"{output_path}.csv"
    pandas_df.to_csv(final_path, index=False)
    print(f" Pipeline Output successfully saved to {final_path}")
    
def save_output_to_mongodb(df):
    """
    Saves the predictive hotspot results directly to MongoDB Atlas.
    """
    print("Connecting to MongoDB Atlas and writing results...")
    
    # ML vectors (like "features") cannot be saved in MongoDB easily, 
    # so we drop them before saving.
    columns_to_drop = ["features", "rawPrediction", "probability"]
    final_df = df.drop(*[c for c in columns_to_drop if c in df.columns])

    # Write to MongoDB
    final_df.write.format("com.mongodb.spark.sql.DefaultSource") \
        .mode("overwrite") \
        .save()
        
    print("✅ Successfully saved AI results to MongoDB Atlas!")
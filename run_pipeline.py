from spark_pipeline.load_data import load_crime_data
from spark_pipeline.preprocessing import preprocess_data
from spark_pipeline.feature_engineering import create_features
from models.hotspot_prediction import detect_hotspots
from models.random_forest import train_random_forest

spark, df = load_crime_data()

df_clean = preprocess_data(df)

rf_predictions = train_random_forest(df_clean)

rf_predictions.show()

features = create_features(df_clean)

hotspots = detect_hotspots(features)

hotspots.show()
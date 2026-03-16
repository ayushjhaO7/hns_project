from spark_pipeline.load_data import load_crime_data
from spark_pipeline.preprocessing import preprocess_data
from spark_pipeline.feature_engineering import create_features

from models.prepare_features import prepare_ml_features
from models.random_forest import random_forest_model
from models.gradient_boosting import gradient_boosting_model
from models.hotspot_prediction import detect_hotspots

from visualization.crime_trend import plot_crime_trend
from visualization.crime_heatmap import generate_heatmap

spark, df = load_crime_data()

df_clean = preprocess_data(df)

plot_crime_trend(df_clean)

features = create_features(df_clean)

ml_data = prepare_ml_features(features)

rf_pred = random_forest_model(ml_data)

hotspots = detect_hotspots(features)

hotspots.show()

generate_heatmap()
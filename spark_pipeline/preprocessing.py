from pyspark.sql.functions import col, upper

def clean_data(df):
    critical_columns = ["STATE/UT", "DISTRICT", "crime_type", "crime_count"]
    existing_cols = [c for c in critical_columns if c in df.columns]
    
    df_clean = df.dropna(subset=existing_cols)
    
    # --- THE FIX ---
    # Remove any rows where the District is a "TOTAL" summary row
    df_clean = df_clean.filter(~upper(col("DISTRICT")).like("%TOTAL%"))
    # ----------------
    
    df_clean = df_clean.dropDuplicates()
    
    return df_clean
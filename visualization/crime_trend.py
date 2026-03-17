import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_extra_plots(raw_data_path="data/crime_dataset.csv", 
                         predicted_path="data/predicted_hotspots.csv"):
    
    if not os.path.exists(raw_data_path) or not os.path.exists(predicted_path):
        print("Error: Required CSV files not found.")
        return

    # Load Data
    df_raw = pd.read_csv(raw_data_path)
    df_pred = pd.read_csv(predicted_path)
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: YEARLY TREND ---
    plt.figure(figsize=(10, 6))
    yearly_data = df_raw.groupby('YEAR')['crime_count'].sum().reset_index()
    sns.lineplot(data=yearly_data, x='YEAR', y='crime_count', marker='o', color='teal', linewidth=2.5)
    plt.title("Yearly Total Crime Trend in India", fontsize=15)
    plt.ylabel("Total Crime Count")
    plt.xlabel("Year")
    plt.savefig("yearly_crime_trend.png")
    print("Generated: yearly_crime_trend.png")

    # --- PLOT 2: CRIME TYPE BREAKDOWN ---
    plt.figure(figsize=(12, 8))
    type_data = df_raw.groupby('crime_type')['crime_count'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=type_data.values, y=type_data.index, palette="viridis")
    plt.title("Top 10 Most Frequent Crime Types", fontsize=15)
    plt.xlabel("Total Incidents")
    plt.ylabel("Crime Category")
    plt.tight_layout()
    plt.savefig("crime_type_breakdown.png")
    print("Generated: crime_type_breakdown.png")

    # --- PLOT 3: STATE SEVERITY RANKING ---
    plt.figure(figsize=(12, 10))
    # Calculate average risk level per state from your LR predictions
    state_severity = df_pred.groupby('STATE/UT')['predicted_risk_level'].mean().sort_values(ascending=False)
    sns.barplot(x=state_severity.values, y=state_severity.index, palette="Reds_r")
    plt.title("State-wise Average Crime Risk Severity (Predicted by AI)", fontsize=15)
    plt.xlabel("Average Risk Level (0-4)")
    plt.ylabel("State / Union Territory")
    plt.tight_layout()
    plt.savefig("state_risk_severity.png")
    print("Generated: state_risk_severity.png")

if __name__ == "__main__":
    generate_extra_plots()
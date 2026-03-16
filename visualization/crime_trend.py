import pandas as pd
import matplotlib.pyplot as plt


def plot_crime_trend(spark_df):

    # Convert Spark DataFrame → Pandas
    df = spark_df.toPandas()

    # Aggregate crime by year
    trend = df.groupby("YEAR")["crime_count"].sum().reset_index()

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(trend["YEAR"], trend["crime_count"], marker='o')

    plt.title("Crime Trend Over Years")
    plt.xlabel("Year")
    plt.ylabel("Total Crime")
    plt.grid(True)

    plt.savefig("crime_trend.png")

    plt.show()
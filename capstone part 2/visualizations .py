#!/usr/bin/env python
# coding: utf-8

# In[12]:


import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger for this module
logger = logging.getLogger(__name__)


def generate_traffic_visualizations(df):
    """Task 3: Create at least three Matplotlib visualisations,

    save them as image files, display them inline, and log each successful save
    with its file path.
    """
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # ==========================================
    # VISUALIZATION 1: Traffic Demand by Hour
    # ==========================================
    plt.figure(figsize=(10, 5))
    hourly_traffic = df.groupby("hour")["traffic_volume"].mean().reset_index()
    sns.lineplot(
        data=hourly_traffic,
        x="hour",
        y="traffic_volume",
        marker="o",
        color="b",
        linewidth=2,
    )
    plt.title("Average Traffic Demand by Hour of the Day")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Average Traffic Volume")
    plt.grid(True, linestyle="--", alpha=0.6)

    path_1 = os.path.join("plots", "traffic_by_hour.png")
    plt.savefig(path_1, bbox_inches="tight")
    plt.show()  # Display inline
    plt.close()

    # Log successful save with file path (INFO level requirement)
    logger.info(f"Figure successfully saved to {path_1}")

    # ==========================================
    # VISUALIZATION 2: Weekday vs. Weekend Traffic
    # ==========================================
    plt.figure(figsize=(8, 5))
    df["day_type"] = df["is_weekend"].apply(
        lambda x: "Weekend" if x == 1 else "Weekday"
    )
    sns.barplot(
        data=df, x="day_type", y="traffic_volume", errorbar=None, palette="Set2"
    )
    plt.title("Traffic Volume: Weekday vs. Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Average Traffic Volume")

    path_2 = os.path.join("plots", "weekday_vs_weekend.png")
    plt.savefig(path_2, bbox_inches="tight")
    plt.show()  # Display inline
    plt.close()

    # Log successful save with file path (INFO level requirement)
    logger.info(f"Figure successfully saved to {path_2}")

    # ==========================================
    # VISUALIZATION 3: Temperature vs. Traffic
    # ==========================================
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df.sample(min(len(df), 2000)),
        x="temp",
        y="traffic_volume",
        alpha=0.5,
        color="purple",
    )
    plt.title("Temperature vs. Traffic Volume")
    plt.xlabel("Temperature (Kelvin / Scaled)")
    plt.ylabel("Traffic Volume")
    plt.grid(True, linestyle="--", alpha=0.6)

    path_3 = os.path.join("plots", "temperature_vs_traffic.png")
    plt.savefig(path_3, bbox_inches="tight")
    plt.show()  # Display inline
    plt.close()

    # Log successful save with file path (INFO level requirement)
    logger.info(f"Figure successfully saved to {path_3}")

    # Log completion using INFO instead of print
    logger.info("All 3 visualizations generated, saved, displayed, and logged successfully!")


# Execute the visualization function
if __name__ == "__main__":
    # Assuming df_engineered is available from previous tasks
    generate_traffic_visualizations(df_engineered)


# 1. For traffic_by_hour.png (Average Traffic Demand by Hour of the Day):
# Interpretation: The line plot clearly illustrates distinct commuter patterns throughout the day. Traffic volume remains minimal during the late-night and early-morning hours (12:00 AM – 4:00 AM) and sharply increases starting at 5:00 AM. It demonstrates a major morning peak around 7:00 AM and reaches its absolute peak during the late afternoon/evening rush hours (around 4:00 PM – 5:00 PM), before gradually declining into the night.
# 
# 2. For weekday_vs_weekend.png (Traffic Volume: Weekday vs. Weekend):
# Interpretation: The bar chart provides a direct comparison between weekday and weekend traffic volumes. It shows that weekdays experience significantly higher average traffic density (exceeding 3,500 vehicles) compared to weekends (which average around 2,500 vehicles). This reflects regular workweek commuting, school runs, and commercial transport activities that subside during weekends.
# 
# 3. For temperature_vs_traffic.png (Temperature vs. Traffic Volume):
# Interpretation: The scatter plot displays the distribution of traffic volume across various temperature ranges (approximately 245K to 310K). Data points are spread across different temperature levels without an extreme linear correlation, indicating that while traffic occurs consistently across varying weather conditions, volume density fluctuates based on overall road conditions rather than temperature alone.

# In[ ]:





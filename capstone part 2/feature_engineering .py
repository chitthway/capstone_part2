#!/usr/bin/env python
# coding: utf-8

# In[9]:


import logging
import os
import numpy as np
import pandas as pd

# 1. CONFIGURE LOGGING
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler and console handler
file_handler = logging.FileHandler("pipeline_and_features.log")
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)  # Console shows INFO and above (hides DEBUG unless configured)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# --- TASK 1 FUNCTIONS (Reference) ---
def load_data(file_path):
    try:
        logger.info(f"Attempting to load dataset from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded raw file. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}", exc_info=True)
        raise


def validate_schema(df):
    expected_columns = [
        "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
        "weather_main", "weather_description", "date_time", "traffic_volume"
    ]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Schema validation failed! Missing columns: {missing_cols}", exc_info=True)
        raise ValueError(f"Missing expected columns: {missing_cols}")
    logger.info("Schema validation passed successfully. All columns present.")


def clean_data(df):
    df = df.copy()
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate rows from the dataset.")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    invalid_dates = df["date_time"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Dropped {invalid_dates} rows due to unparseable date_time values.")
        df = df.dropna(subset=["date_time"])

    df["month"] = df["date_time"].dt.month

    # Impute temperature outliers (<= 0 Kelvin) using monthly median
    temp_outliers = df["temp"] <= 0
    if temp_outliers.sum() > 0:
        count = temp_outliers.sum()
        for month in df["month"].unique():
            month_median = df.loc[(df["month"] == month) & (df["temp"] > 0), "temp"].median()
            df.loc[(df["month"] == month) & (df["temp"] <= 0), "temp"] = month_median
        logger.warning(f"Imputed {count} temperature outliers (<= 0 Kelvin) using monthly median values.")

    # Impute rain outliers (> 9000 mm) using global median
    rain_outliers = df["rain_1h"] > 9000
    if rain_outliers.sum() > 0:
        count = rain_outliers.sum()
        global_rain_median = df.loc[df["rain_1h"] <= 9000, "rain_1h"].median()
        df.loc[df["rain_1h"] > 9000, "rain_1h"] = global_rain_median
        logger.warning(f"Imputed {count} impossible rain records (> 9000 mm) using the median value.")

    logger.info("Data cleaning and outlier imputation completed successfully.")
    return df


# --- TASK 2 FUNCTION: FEATURE ENGINEERING ---
def engineer_features(df):
    """Task 2: Feature Engineering using NumPy and Pandas with required logging.

    Documenting Traffic Target Logic:
    The congestion category is created based on the 25th (Q25) and 75th (Q75) 
    quartiles of the 'traffic_volume' column:
    - 0 (Low): traffic_volume <= Q25
    - 1 (Medium): Q25 < traffic_volume <= Q75
    - 2 (High): traffic_volume > Q75
    """
    df = df.copy()

    # 1. Log dataset shape BEFORE feature engineering (INFO level)
    logger.info(
        f"Dataset shape BEFORE feature engineering - Rows: {df.shape[0]}, "
        f"Columns: {df.shape[1]}"
    )

    # 2. Time Features & Cyclical Encoding (NumPy sin/cos)
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # 3. Weather Categorical Encoding
    if "weather_main" in df.columns:
        df = pd.get_dummies(df, columns=["weather_main"], prefix="weather", drop_first=True)

    # 4. Numerical Scaling (Normalised versions of continuous variables: temp, clouds_all)
    for col in ["temp", "clouds_all"]:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            # Log intermediate scaling parameters at DEBUG level
            logger.debug(f"Scaling continuous variable '{col}': min={min_val}, max={max_val}")
            df[f"{col}_scaled"] = (df[col] - min_val) / (max_val - min_val)

    # 5. Traffic Target Congestion Category based on Quartiles
    q25 = df["traffic_volume"].quantile(0.25)
    q75 = df["traffic_volume"].quantile(0.75)

    # Log intermediate threshold calculations at DEBUG level
    logger.debug(
        f"Traffic congestion logic - Quartile thresholds calculated: Q25={q25}, Q75={q75}"
    )

    def categorize_traffic(vol):
        if vol <= q25:
            return 0  # Low
        elif vol <= q75:
            return 1  # Medium
        else:
            return 2  # High

    df["congestion_category"] = df["traffic_volume"].apply(categorize_traffic)

    # 6. Log dataset shape AFTER feature engineering (INFO level)
    logger.info(
        f"Dataset shape AFTER feature engineering - Rows: {df.shape[0]}, "
        f"Columns: {df.shape[1]}"
    )

    return df


# --- EXECUTION & TESTING BLOCK (No print statements for internal status) ---
if __name__ == "__main__":
    desktop_path = os.path.expanduser("~/Desktop")
    csv_file = os.path.join(desktop_path, "Metro_Interstate_Traffic_Volume.csv")

    try:
        # Run Task 1 Pipeline
        df_raw = load_data(csv_file)
        validate_schema(df_raw)
        df_cleaned = clean_data(df_raw)

        # Run Task 2 Feature Engineering
        df_engineered = engineer_features(df_cleaned)

        # Log completion milestones using INFO level instead of print()
        logger.info("Pipeline and Feature Engineering workflow completed successfully!")
        logger.info(f"Final Engineered Dataset Shape: {df_engineered.shape}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)


# In[ ]:





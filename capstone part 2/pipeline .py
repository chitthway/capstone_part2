#!/usr/bin/env python
# coding: utf-8

# task 1

# In[4]:


import logging
import os
import pandas as pd

# 1. CONFIGURE LOGGING (Writes to both console and 'pipeline.log' file)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler and console handler
file_handler = logging.FileHandler("pipeline.log")
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Create formatter and add it to handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_data(file_path):
  """Task 1.1: Load raw CSV file with robust exception handling."""
  try:
    logger.info(f"Attempting to load dataset from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(
        f"Successfully loaded raw file. Rows: {df.shape[0]}, Columns:"
        f" {df.shape[1]}"
    )
    return df
  except FileNotFoundError as e:
    logger.error(
        f"File not found error: {file_path}. Please check the path.",
        exc_info=True,
    )
    raise
  except Exception as e:
    logger.error(
        f"An unexpected error occurred while loading CSV: {e}", exc_info=True
    )
    raise


def validate_schema(df):
  """Task 1.2: Validate that expected columns are present."""
  expected_columns = [
      "holiday",
      "temp",
      "rain_1h",
      "snow_1h",
      "clouds_all",
      "weather_main",
      "weather_description",
      "date_time",
      "traffic_volume",
  ]
  missing_cols = [col for col in expected_columns if col not in df.columns]

  if missing_cols:
    logger.error(
        f"Schema validation failed! Missing columns: {missing_cols}",
        exc_info=True,
    )
    raise ValueError(f"Missing expected columns: {missing_cols}")

  logger.info("Schema validation passed successfully. All columns present.")


def clean_data(df):
  """Task 1.3 & 1.4: Clean data, handle duplicates, outliers, and imputation using loops/conditionals."""
  df = df.copy()
  initial_rows = len(df)

  # 1. Remove Duplicates
  df = df.drop_duplicates()
  duplicates_removed = initial_rows - len(df)
  if duplicates_removed > 0:
    logger.warning(
        f"Removed {duplicates_removed} duplicate rows from the dataset."
    )
  else:
    logger.info("No duplicate rows found.")

  # 2. Parse and Validate Date/Time
  df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
  invalid_dates = df["date_time"].isna().sum()
  if invalid_dates > 0:
    logger.warning(
        f"Dropped {invalid_dates} rows due to unparseable date_time values."
    )
    df = df.dropna(subset=["date_time"])

  # Extract month for group-by operations
  df["month"] = df["date_time"].dt.month

  # 3. Detect and Impute Outliers
  temp_outliers = df["temp"] <= 0
  if temp_outliers.sum() > 0:
    count = temp_outliers.sum()
    for month in df["month"].unique():
      month_median = df.loc[
          (df["month"] == month) & (df["temp"] > 0), "temp"
      ].median()
      df.loc[(df["month"] == month) & (df["temp"] <= 0), "temp"] = month_median
    logger.warning(
        f"Imputed {count} temperature outliers (<= 0 Kelvin) using monthly median"
        " values."
    )

  rain_outliers = df["rain_1h"] > 9000
  if rain_outliers.sum() > 0:
    count = rain_outliers.sum()
    global_rain_median = df.loc[df["rain_1h"] <= 9000, "rain_1h"].median()
    df.loc[df["rain_1h"] > 9000, "rain_1h"] = global_rain_median
    logger.warning(
        f"Imputed {count} impossible rain records (> 9000 mm) using the median"
        " value."
    )

  logger.info("Data cleaning and outlier imputation completed successfully.")
  return df


if __name__ == "__main__":
  try:
    logger.info("Starting pipeline execution...")
    desktop_path = os.path.expanduser("~/Desktop")
    csv_file = os.path.join(
        desktop_path, "Metro_Interstate_Traffic_Volume.csv"
    )

    df_raw = load_data(csv_file)
    validate_schema(df_raw)
    df_cleaned = clean_data(df_raw)

    logger.info("Pipeline Task 1 finished successfully!")
  except Exception as e:
    logger.error("Pipeline failed to complete execution.", exc_info=True)


# In[ ]:





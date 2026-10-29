#!/usr/bin/env python
# coding: utf-8

# In[13]:


import logging
import os
import pandas as pd

# 1. CONFIGURE LOGGING (Following project logging requirements)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler and console handler
file_handler = logging.FileHandler("pipeline.log")
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_processed_data():
    """Load the processed dataset for the CLI application."""
    # Assuming cleaned/engineered data or csv can be loaded
    desktop_path = os.path.expanduser("~/Desktop")
    csv_file = os.path.join(desktop_path, "Metro_Interstate_Traffic_Volume.csv")

    try:
        logger.info(f"CLI application attempting to load dataset from {csv_file}")
        df = pd.read_csv(csv_file)
        # Parse date_time for date-based queries
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        logger.info("Dataset successfully loaded for CLI application.")
        return df
    except Exception as e:
        logger.error("Failed to load dataset for CLI application.", exc_info=True)
        raise


def run_cli():
    """Task 4: Simple command-line application supporting at least 3 queries."""
    df = load_processed_data()

    print("\n==================================================")
    print("🚗 WELCOME TO THE TRAFFIC ANALYTICS CLI APPLICATION 🚗")
    print("==================================================")
    print("Available Commands / Queries:")
    print("  1. query_date <YYYY-MM-DD HH:MM> - Check traffic for a specific date/time")
    print("  2. high_traffic                   - Identify high-traffic periods")
    print("  3. compare_days                   - Compare weekday vs weekend traffic")
    print("  4. exit                           - Exit the application")
    print("--------------------------------------------------\n")

    while True:
        try:
            user_input = input("Enter command (e.g., query_date 2012-10-02 09:00): ").strip()

            if not user_input:
                continue

            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "exit":
                logger.info("User invoked 'exit' command. Closing CLI application.")
                print("Exiting application. Thank you!")
                break

            elif command == "query_date":
                logger.info(f"Command invoked: 'query_date' with arguments: '{args}'")
                if not args:
                    raise ValueError("Missing date/time argument. Format required: YYYY-MM-DD HH:MM")

                # Validate input date format
                query_dt = pd.to_datetime(args, errors="raise")
                match = df[df["date_time"] == query_dt]

                if not match.empty:
                    traffic_vol = match["traffic_volume"].values[0]
                    temp_val = match["temp"].values[0]
                    weather = match["weather_main"].values[0] if "weather_main" in match.columns else "N/A"

                    # print() is permitted for final user output in CLI
                    print(f"\n[Result] Date/Time: {query_dt}")
                    print(f"  - Traffic Volume: {traffic_vol} vehicles")
                    print(f"  - Temperature: {temp_val}")
                    print(f"  - Weather: {weather}\n")
                else:
                    print(f"\n[Result] No traffic record found for exact timestamp: {query_dt}\n")

            elif command == "high_traffic":
                logger.info("Command invoked: 'high_traffic'")
                q90 = df["traffic_volume"].quantile(0.90)
                high_df = df[df["traffic_volume"] >= q90]

                print(f"\n[Result] High-Traffic Periods (Threshold >= 90th percentile: {q90:.1f} vehicles):")
                print(f"  - Total high-traffic records found: {len(high_df)}")
                print(f"  - Average temperature during high traffic: {high_df['temp'].mean():.2f}\n")

            elif command == "compare_days":
                logger.info("Command invoked: 'compare_days'")
                df["is_weekend"] = df["date_time"].dt.dayofweek.isin([5, 6])
                comparison = df.groupby("is_weekend")["traffic_volume"].mean()

                weekday_avg = comparison.get(False, 0)
                weekend_avg = comparison.get(True, 0)

                print(f"\n[Result] Weekday vs Weekend Traffic Comparison:")
                print(f"  - Weekday Average Traffic Volume: {weekday_avg:.2f} vehicles")
                print(f"  - Weekend Average Traffic Volume: {weekend_avg:.2f} vehicles\n")

            else:
                logger.warning(f"User entered unknown command: '{command}'")
                print(f"Unknown command '{command}'. Please choose from: query_date, high_traffic, compare_days, exit.")

        except ValueError as ve:
            # Log clear ERROR instead of raw traceback when user supplies invalid input
            logger.error(f"Invalid user input supplied: '{user_input}'. Error: {ve}", exc_info=False)
            print(f"\n[Error] Invalid input or malformed date. Please check your format (e.g., YYYY-MM-DD HH:MM).\n")
        except Exception as e:
            logger.error(f"An unexpected error occurred in CLI: {e}", exc_info=True)
            print(f"\n[Error] An unexpected error occurred. Please try again.\n")


if __name__ == "__main__":
    run_cli()


# In[ ]:





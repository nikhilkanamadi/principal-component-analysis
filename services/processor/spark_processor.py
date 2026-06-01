"""
PySpark Feature Engineering Processor

Reads raw_weather from PostgreSQL, applies feature engineering
using Spark SQL window functions, and writes back to processed_weather.

Run:
  spark-submit --master spark://spark-master:7077 \
      services/processor/spark_processor.py \
      --location "London"

Or locally (no cluster):
  python -m services.processor.spark_processor --location "London"
"""

import os
import json
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATABASE_URL  = os.getenv("DATABASE_URL", "sqlite:///./data/weather.db")
SPARK_MASTER  = os.getenv("SPARK_MASTER", "local[*]")
PG_JDBC_URL   = os.getenv("PG_JDBC_URL", "")
PG_USER       = os.getenv("POSTGRES_USER", "weather")
PG_PASSWORD   = os.getenv("POSTGRES_PASSWORD", "weather123")


def _get_spark():
    from pyspark.sql import SparkSession
    builder = (
        SparkSession.builder
        .appName("WeatherFeatureEngineering")
        .master(SPARK_MASTER)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
    )
    if PG_JDBC_URL:
        builder = builder.config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
    return builder.getOrCreate()


def process_with_spark(location: str) -> int:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import DoubleType

    spark = _get_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ---- Load raw data ----
    if PG_JDBC_URL:
        df = (
            spark.read.format("jdbc")
            .option("url", PG_JDBC_URL)
            .option("dbtable", f"(SELECT * FROM raw_weather WHERE location='{location}') t")
            .option("user", PG_USER)
            .option("password", PG_PASSWORD)
            .option("driver", "org.postgresql.Driver")
            .load()
        )
    else:
        # Fallback: load via pandas from SQLAlchemy
        import pandas as pd
        from app.core.storage import load_raw
        pdf = load_raw(location)
        if pdf.empty:
            log.warning("No raw data found for '%s'", location)
            return 0
        df = spark.createDataFrame(pdf.astype(str))

    df = df.withColumn("date", F.to_date("date"))

    numeric_cols = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum",  "wind_speed_10m_max", "wind_gusts_10m_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    ]
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # ---- Feature engineering ----
    # Derived features
    df = (df
        .withColumn("temp_range",     F.col("temperature_2m_max") - F.col("temperature_2m_min"))
        .withColumn("wind_gust_ratio", F.col("wind_gusts_10m_max") / (F.col("wind_speed_10m_max") + 1e-6))
        .withColumn("precip_flag",    (F.col("precipitation_sum") > 0).cast(DoubleType()))
        .withColumn("day_of_year",    F.dayofyear("date").cast(DoubleType()))
        .withColumn("month",          F.month("date").cast(DoubleType()))
    )

    # Rolling 7-day window (unix_date gives integer days since epoch — safe for ordering in Spark 3.5+)
    df = df.withColumn("_date_int", F.unix_date("date"))
    w7 = (Window.partitionBy("location")
              .orderBy("_date_int")
              .rowsBetween(-6, 0))

    for col in ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]:
        df = (df
            .withColumn(f"{col}_roll7_mean", F.avg(col).over(w7))
            .withColumn(f"{col}_roll7_std",  F.stddev(col).over(w7))
        )

    # Deviation from rolling mean
    df = (df
        .withColumn("temp_deviation",
                    F.col("temperature_2m_mean") - F.col("temperature_2m_mean_roll7_mean"))
        .withColumn("precip_deviation",
                    F.col("precipitation_sum") - F.col("precipitation_sum_roll7_mean"))
        .fillna(0.0)
        .drop("_date_int")
    )

    feature_cols = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum",  "wind_speed_10m_max", "wind_gusts_10m_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration",
        "temp_range", "wind_gust_ratio", "precip_flag",
        "temperature_2m_mean_roll7_mean", "temperature_2m_mean_roll7_std",
        "precipitation_sum_roll7_mean",   "precipitation_sum_roll7_std",
        "wind_speed_10m_max_roll7_mean",  "wind_speed_10m_max_roll7_std",
        "temp_deviation", "precip_deviation",
        "day_of_year", "month",
    ]

    result = df.select(["location", "date"] + feature_cols)

    # ---- Write back ----
    if PG_JDBC_URL:
        result.write.format("jdbc") \
            .option("url", PG_JDBC_URL) \
            .option("dbtable", "processed_weather_spark") \
            .option("user", PG_USER) \
            .option("password", PG_PASSWORD) \
            .option("driver", "org.postgresql.Driver") \
            .mode("overwrite") \
            .save()
        count = result.count()
    else:
        from app.core.storage import save_processed
        import pandas as pd
        pdf = result.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        count = save_processed(pdf, location)

    log.info("Spark processed %d records for '%s'", count, location)
    spark.stop()
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="London")
    args = parser.parse_args()
    process_with_spark(args.location)

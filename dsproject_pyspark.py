from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, isnan, isnull, when, count, sum as spark_sum, 
    to_timestamp, hour, minute, split, regexp_extract,
    desc, asc, collect_list, size, explode, combinations
)
from pyspark.sql.types import IntegerType, FloatType
import matplotlib.pyplot as plt
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("DSProject Analysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Set log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# Merging 12 months of data to a single file
path = "./dsproject sales data"
files = [file for file in os.listdir(path) if not file.startswith('.')]

# Read all CSV files and union them
all_months_data = None
for file in files:
    current_data = spark.read.option("header", "true").csv(f"{path}/{file}")
    if all_months_data is None:
        all_months_data = current_data
    else:
        all_months_data = all_months_data.union(current_data)

# Write to CSV
all_months_data.coalesce(1).write.mode("overwrite").option("header", "true").csv("all_data_spark")

# Read the consolidated data
all_data = spark.read.option("header", "true").csv("all_data_copy.csv")

print("Data preview:")
all_data.show(5)

# Clean up the data - Drop rows of NaN
print("Finding NaN values:")
nan_df = all_data.filter(
    col("Order ID").isNull() | 
    col("Product").isNull() | 
    col("Quantity Ordered").isNull() | 
    col("Price Each").isNull() | 
    col("Order Date").isNull() | 
    col("Purchase Address").isNull()
)
nan_df.show(5)

# Drop rows where all values are null
all_data = all_data.filter(~(
    col("Order ID").isNull() & 
    col("Product").isNull() & 
    col("Quantity Ordered").isNull() & 
    col("Price Each").isNull() & 
    col("Order Date").isNull() & 
    col("Purchase Address").isNull()
))

# Get rid of text in order date column (filter out rows starting with 'Or')
all_data = all_data.filter(~col("Order Date").startswith("Or"))

# Make columns correct type
all_data = all_data.withColumn("Quantity Ordered", col("Quantity Ordered").cast(IntegerType()))
all_data = all_data.withColumn("Price Each", col("Price Each").cast(FloatType()))

# Add month column
all_data = all_data.withColumn("Month", col("Order Date").substr(1, 2).cast(IntegerType()))

# Add month column (alternative method using to_timestamp)
all_data = all_data.withColumn("Month 2", 
    month(to_timestamp(col("Order Date"), "MM/dd/yy HH:mm")))

# Add Sales column
all_data = all_data.withColumn("Sales", col("Quantity Ordered") * col("Price Each"))

# Add City column
all_data = all_data.withColumn("City_State", split(col("Purchase Address"), ","))
all_data = all_data.withColumn("City", 
    concat(
        trim(col("City_State").getItem(1)), 
        lit("  ("), 
        split(trim(col("City_State").getItem(2)), " ").getItem(1), 
        lit(")")
    )
)

print("Data after cleaning and augmentation:")
all_data.show(5)

# Data Exploration
print("\n=== Question 1: What was the best month for sales? ===")
monthly_sales = all_data.groupBy("Month").agg(spark_sum("Sales").alias("Total_Sales"))
monthly_sales.orderBy(desc("Total_Sales")).show()

# Convert to Pandas for plotting (if needed)
monthly_sales_pd = monthly_sales.toPandas()
plt.figure(figsize=(10, 6))
plt.bar(monthly_sales_pd["Month"], monthly_sales_pd["Total_Sales"])
plt.xlabel("Month")
plt.ylabel("Sales in USD ($)")
plt.title("Monthly Sales")
plt.xticks(range(1, 13))
plt.show()

print("\n=== Question 2: What city sold the most product? ===")
city_sales = all_data.groupBy("City").agg(spark_sum("Sales").alias("Total_Sales"))
city_sales.orderBy(desc("Total_Sales")).show()

# Convert to Pandas for plotting
city_sales_pd = city_sales.toPandas()
plt.figure(figsize=(12, 6))
plt.bar(city_sales_pd["City"], city_sales_pd["Total_Sales"])
plt.xlabel("City")
plt.ylabel("Sales in USD ($)")
plt.title("Sales by City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n=== Question 3: What time should we display advertisements? ===")
# Add hour and minute columns
all_data = all_data.withColumn("Hour", hour(to_timestamp(col("Order Date"), "MM/dd/yy HH:mm")))
all_data = all_data.withColumn("Minute", minute(to_timestamp(col("Order Date"), "MM/dd/yy HH:mm")))
all_data = all_data.withColumn("Count", lit(1))

hourly_orders = all_data.groupBy("Hour").agg(spark_sum("Count").alias("Order_Count"))
hourly_orders.orderBy("Hour").show()

# Convert to Pandas for plotting
hourly_orders_pd = hourly_orders.toPandas()
plt.figure(figsize=(10, 6))
plt.plot(hourly_orders_pd["Hour"], hourly_orders_pd["Order_Count"])
plt.xlabel("Hour")
plt.ylabel("Number of Orders")
plt.title("Orders by Hour of Day")
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()

print("\n=== Question 4: What products are most often sold together? ===")
# Group by Order ID to find products sold together
order_products = all_data.groupBy("Order ID").agg(
    collect_list("Product").alias("Products")
).filter(size(col("Products")) > 1)

# This is a simplified approach - for full combinations analysis, 
# you would need to use UDFs or more complex operations
print("Orders with multiple products:")
order_products.show(10, truncate=False)

print("\n=== Question 5: What product sold the most? ===")
product_sales = all_data.groupBy("Product").agg(
    spark_sum("Quantity Ordered").alias("Total_Quantity"),
    spark_sum("Sales").alias("Total_Sales"),
    avg("Price Each").alias("Avg_Price")
)
product_sales.orderBy(desc("Total_Quantity")).show()

# Convert to Pandas for plotting
product_sales_pd = product_sales.toPandas()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Quantity plot
ax1.bar(product_sales_pd["Product"], product_sales_pd["Total_Quantity"])
ax1.set_ylabel("Quantity Ordered")
ax1.set_title("Product Quantity vs Price Analysis")
ax1.tick_params(axis='x', rotation=45)

# Price plot
ax2.bar(product_sales_pd["Product"], product_sales_pd["Avg_Price"])
ax2.set_ylabel("Average Price ($)")
ax2.set_xlabel("Product")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Stop Spark session
spark.stop()
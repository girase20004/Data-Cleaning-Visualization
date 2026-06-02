import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("visualizations", exist_ok=True)

# Load Dataset
df = pd.read_csv("SampleSuperstore.csv")

# Display first rows
print(df.head())

# Dataset information
print(df.info())

# Statistical Summary
print(df.describe())

# Dataset Shape
print("Shape:", df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Missing Values
df = df.dropna()

# Duplicate Records
print("\nDuplicate Records:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Outlier Detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Sales'])
plt.title("Sales Outliers")
plt.savefig("visualizations/boxplot.png")
plt.close()

# Remove Outliers
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Sales'] >= lower) &
        (df['Sales'] <= upper)]

# Save Cleaned Dataset
df.to_csv("cleaned_sales_data.csv", index=False)

# Bar Chart
plt.figure(figsize=(10,5))
sns.countplot(x='Category', data=df)
plt.title("Products by Category")
plt.savefig("visualizations/bar_chart.png")
plt.close()

# Histogram
plt.figure(figsize=(10,5))
plt.hist(df['Sales'], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.savefig("visualizations/histogram.png")
plt.close()

# Pie Chart
plt.figure(figsize=(8,8))
df['Category'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.ylabel("")
plt.savefig("visualizations/pie_chart.png")
plt.close()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Quantity',
    y='Sales',
    data=df
)
plt.title("Quantity vs Sales")
plt.savefig("visualizations/scatter_plot.png")
plt.close()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
plt.title("Correlation Heatmap")
plt.savefig("visualizations/heatmap.png")
plt.close()

print("Project Completed Successfully")
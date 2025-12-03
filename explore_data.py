import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('india_housing_prices.csv')

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)

# Basic information
print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n" + "-"*80)
print("COLUMN NAMES AND DATA TYPES")
print("-"*80)
print(df.dtypes)

print("\n" + "-"*80)
print("FIRST 5 ROWS")
print("-"*80)
print(df.head())

print("\n" + "-"*80)
print("BASIC STATISTICS")
print("-"*80)
print(df.describe())

print("\n" + "-"*80)
print("MISSING VALUES")
print("-"*80)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Percentage': missing_percent.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("No missing values found!")

print("\n" + "-"*80)
print("DUPLICATE ROWS")
print("-"*80)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

print("\n" + "-"*80)
print("MEMORY USAGE")
print("-"*80)
print(f"Total memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)

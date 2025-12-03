import pandas as pd

# Load the dataset
df = pd.read_csv('india_housing_prices.csv')

print("\n" + "="*80)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*80)

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    value_counts = df[col].value_counts()
    print(f"  Top 10 values:")
    for idx, (val, count) in enumerate(value_counts.head(10).items(), 1):
        print(f"    {idx}. {val}: {count} ({count/len(df)*100:.2f}%)")

print("\n" + "="*80)
print("NUMERICAL FEATURES SUMMARY")
print("="*80)

numerical_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 
                  'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
                  'Nearby_Schools', 'Nearby_Hospitals']

for col in numerical_cols:
    print(f"\n{col}:")
    print(f"  Min: {df[col].min()}")
    print(f"  Max: {df[col].max()}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"Price Range: ₹{df['Price_in_Lakhs'].min():.2f}L - ₹{df['Price_in_Lakhs'].max():.2f}L")
print(f"Average Price: ₹{df['Price_in_Lakhs'].mean():.2f}L")
print(f"Median Price: ₹{df['Price_in_Lakhs'].median():.2f}L")
print(f"Average Price per SqFt: ₹{df['Price_per_SqFt'].mean():.2f}")
print(f"Property Age Range: {df['Age_of_Property'].min()}-{df['Age_of_Property'].max()} years")
print(f"Number of States: {df['State'].nunique()}")
print(f"Number of Cities: {df['City'].nunique()}")
print(f"Number of Localities: {df['Locality'].nunique()}")

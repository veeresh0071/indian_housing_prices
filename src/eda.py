import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('india_housing_prices.csv')

# Create outputs directory if it doesn't exist
import os
os.makedirs('outputs', exist_ok=True)

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS - REAL ESTATE INVESTMENT ADVISOR")
print("="*80)

# ============================================================================
# 1. PRICE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("1. PRICE DISTRIBUTION ANALYSIS")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price distribution
axes[0, 0].hist(df['Price_in_Lakhs'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Property Prices', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Price (in Lakhs)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['Price_in_Lakhs'].mean(), color='red', linestyle='--', label=f'Mean: ₹{df["Price_in_Lakhs"].mean():.2f}L')
axes[0, 0].axvline(df['Price_in_Lakhs'].median(), color='green', linestyle='--', label=f'Median: ₹{df["Price_in_Lakhs"].median():.2f}L')
axes[0, 0].legend()

# Size distribution
axes[0, 1].hist(df['Size_in_SqFt'], bins=50, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('Distribution of Property Sizes', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Size (in Sq Ft)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['Size_in_SqFt'].mean(), color='red', linestyle='--', label=f'Mean: {df["Size_in_SqFt"].mean():.0f} sqft')
axes[0, 1].legend()

# Price per Sq Ft distribution
axes[1, 0].hist(df['Price_per_SqFt'], bins=50, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Distribution of Price per Sq Ft', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Price per Sq Ft')
axes[1, 0].set_ylabel('Frequency')

# BHK distribution
bhk_counts = df['BHK'].value_counts().sort_index()
axes[1, 1].bar(bhk_counts.index, bhk_counts.values, color='plum', edgecolor='black')
axes[1, 1].set_title('Distribution of BHK', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Number of Bedrooms (BHK)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/01_price_size_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/01_price_size_distribution.png")
plt.close()

# ============================================================================
# 2. PRICE ANALYSIS BY LOCATION
# ============================================================================
print("\n" + "-"*80)
print("2. PRICE ANALYSIS BY LOCATION")
print("-"*80)

# Average price by state
state_price = df.groupby('State')['Price_in_Lakhs'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
print("\nTop 10 States by Average Price:")
print(state_price.head(10))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Top 10 states by average price
top_states = state_price.head(10)
axes[0].barh(range(len(top_states)), top_states['mean'], color='steelblue')
axes[0].set_yticks(range(len(top_states)))
axes[0].set_yticklabels(top_states.index)
axes[0].set_xlabel('Average Price (in Lakhs)')
axes[0].set_title('Top 10 States by Average Property Price', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Top 15 cities by average price
city_price = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15)
axes[1].barh(range(len(city_price)), city_price.values, color='coral')
axes[1].set_yticks(range(len(city_price)))
axes[1].set_yticklabels(city_price.index)
axes[1].set_xlabel('Average Price (in Lakhs)')
axes[1].set_title('Top 15 Cities by Average Property Price', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/02_price_by_location.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/02_price_by_location.png")
plt.close()

# ============================================================================
# 3. PRICE ANALYSIS BY PROPERTY CHARACTERISTICS
# ============================================================================
print("\n" + "-"*80)
print("3. PRICE ANALYSIS BY PROPERTY CHARACTERISTICS")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Price by Property Type
property_type_price = df.groupby('Property_Type')['Price_in_Lakhs'].mean().sort_values(ascending=False)
axes[0, 0].bar(property_type_price.index, property_type_price.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Average Price by Property Type', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Average Price (in Lakhs)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Price by BHK
bhk_price = df.groupby('BHK')['Price_in_Lakhs'].mean()
axes[0, 1].plot(bhk_price.index, bhk_price.values, marker='o', linewidth=2, markersize=10, color='#FF6B6B')
axes[0, 1].set_title('Average Price by BHK', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Bedrooms (BHK)')
axes[0, 1].set_ylabel('Average Price (in Lakhs)')
axes[0, 1].grid(True, alpha=0.3)

# Price by Furnished Status
furnished_price = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False)
axes[1, 0].bar(furnished_price.index, furnished_price.values, color=['#95E1D3', '#F38181', '#FEC260'])
axes[1, 0].set_title('Average Price by Furnished Status', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Price (in Lakhs)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Price by Availability Status
availability_price = df.groupby('Availability_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False)
axes[1, 1].bar(availability_price.index, availability_price.values, color=['#A8E6CF', '#FFD3B6'])
axes[1, 1].set_title('Average Price by Availability Status', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Average Price (in Lakhs)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/03_price_by_characteristics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/03_price_by_characteristics.png")
plt.close()

# ============================================================================
# 4. SIZE VS PRICE RELATIONSHIP
# ============================================================================
print("\n" + "-"*80)
print("4. SIZE VS PRICE RELATIONSHIP")
print("-"*80)

# Calculate correlation
correlation = df['Size_in_SqFt'].corr(df['Price_in_Lakhs'])
print(f"\nCorrelation between Size and Price: {correlation:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot: Size vs Price
sample = df.sample(n=5000, random_state=42)  # Sample for better visualization
axes[0].scatter(sample['Size_in_SqFt'], sample['Price_in_Lakhs'], alpha=0.3, c='steelblue', s=10)
axes[0].set_title('Property Size vs Price (5,000 samples)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Size (in Sq Ft)')
axes[0].set_ylabel('Price (in Lakhs)')
axes[0].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Box plot: Price per Sq Ft by Property Type
df.boxplot(column='Price_per_SqFt', by='Property_Type', ax=axes[1])
axes[1].set_title('Price per Sq Ft by Property Type', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Property Type')
axes[1].set_ylabel('Price per Sq Ft')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('outputs/04_size_vs_price.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/04_size_vs_price.png")
plt.close()

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("5. CORRELATION ANALYSIS")
print("-"*80)

# Select numerical columns for correlation
numerical_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 
                  'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
                  'Nearby_Schools', 'Nearby_Hospitals']

correlation_matrix = df[numerical_cols].corr()
print("\nCorrelation with Price_in_Lakhs:")
price_corr = correlation_matrix['Price_in_Lakhs'].sort_values(ascending=False)
print(price_corr)

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/05_correlation_heatmap.png")
plt.close()

# ============================================================================
# 6. IMPACT OF AMENITIES AND INFRASTRUCTURE
# ============================================================================
print("\n" + "-"*80)
print("6. IMPACT OF AMENITIES AND INFRASTRUCTURE")
print("-"*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Parking Space impact
parking_price = df.groupby('Parking_Space')['Price_in_Lakhs'].mean()
axes[0, 0].bar(parking_price.index, parking_price.values, color=['#FF6B6B', '#4ECDC4'])
axes[0, 0].set_title('Average Price by Parking Space', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Average Price (in Lakhs)')

# Security impact
security_price = df.groupby('Security')['Price_in_Lakhs'].mean()
axes[0, 1].bar(security_price.index, security_price.values, color=['#FFD93D', '#6BCB77'])
axes[0, 1].set_title('Average Price by Security', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Average Price (in Lakhs)')

# Public Transport Accessibility
transport_price = df.groupby('Public_Transport_Accessibility')['Price_in_Lakhs'].mean().sort_values(ascending=False)
axes[0, 2].bar(transport_price.index, transport_price.values, color=['#95E1D3', '#F38181', '#FEC260'])
axes[0, 2].set_title('Average Price by Transport Access', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('Average Price (in Lakhs)')

# Nearby Schools impact
schools_price = df.groupby('Nearby_Schools')['Price_in_Lakhs'].mean()
axes[1, 0].plot(schools_price.index, schools_price.values, marker='o', linewidth=2, color='#5D5FEF')
axes[1, 0].set_title('Price vs Nearby Schools', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Nearby Schools')
axes[1, 0].set_ylabel('Average Price (in Lakhs)')
axes[1, 0].grid(True, alpha=0.3)

# Nearby Hospitals impact
hospitals_price = df.groupby('Nearby_Hospitals')['Price_in_Lakhs'].mean()
axes[1, 1].plot(hospitals_price.index, hospitals_price.values, marker='o', linewidth=2, color='#FF6B6B')
axes[1, 1].set_title('Price vs Nearby Hospitals', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Nearby Hospitals')
axes[1, 1].set_ylabel('Average Price (in Lakhs)')
axes[1, 1].grid(True, alpha=0.3)

# Property Age vs Price
age_price = df.groupby('Age_of_Property')['Price_in_Lakhs'].mean()
axes[1, 2].plot(age_price.index, age_price.values, linewidth=2, color='#4ECDC4')
axes[1, 2].set_title('Price vs Property Age', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Age of Property (years)')
axes[1, 2].set_ylabel('Average Price (in Lakhs)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/06_amenities_infrastructure_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/06_amenities_infrastructure_impact.png")
plt.close()

# ============================================================================
# 7. OWNER TYPE AND FACING ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("7. OWNER TYPE AND FACING ANALYSIS")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Owner Type impact
owner_price = df.groupby('Owner_Type')['Price_in_Lakhs'].mean().sort_values(ascending=False)
axes[0].bar(owner_price.index, owner_price.values, color=['#A8E6CF', '#FFD3B6', '#FF8B94'])
axes[0].set_title('Average Price by Owner Type', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Average Price (in Lakhs)')

# Facing direction impact
facing_price = df.groupby('Facing')['Price_in_Lakhs'].mean().sort_values(ascending=False)
axes[1].bar(facing_price.index, facing_price.values, color=['#FFDAC1', '#FF9AA2', '#B5EAD7', '#C7CEEA'])
axes[1].set_title('Average Price by Facing Direction', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Average Price (in Lakhs)')

plt.tight_layout()
plt.savefig('outputs/07_owner_facing_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/07_owner_facing_analysis.png")
plt.close()

# ============================================================================
# 8. TOP AMENITIES ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("8. TOP AMENITIES ANALYSIS")
print("-"*80)

# Get top 15 amenities by frequency
top_amenities = df['Amenities'].value_counts().head(15)
print("\nTop 15 Most Common Amenities:")
print(top_amenities)

# Average price for top amenities
amenities_price = []
for amenity in top_amenities.index:
    avg_price = df[df['Amenities'] == amenity]['Price_in_Lakhs'].mean()
    amenities_price.append(avg_price)

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Frequency of top amenities
axes[0].barh(range(len(top_amenities)), top_amenities.values, color='skyblue')
axes[0].set_yticks(range(len(top_amenities)))
axes[0].set_yticklabels(top_amenities.index)
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top 15 Most Common Amenities', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Average price for top amenities
axes[1].barh(range(len(top_amenities)), amenities_price, color='lightcoral')
axes[1].set_yticks(range(len(top_amenities)))
axes[1].set_yticklabels(top_amenities.index)
axes[1].set_xlabel('Average Price (in Lakhs)')
axes[1].set_title('Average Price for Top 15 Amenities', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/08_top_amenities_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/08_top_amenities_analysis.png")
plt.close()

# ============================================================================
# 9. STATISTICAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)

summary_stats = {
    'Feature': [],
    'Impact': [],
    'Observation': []
}

# Property Type
pt_range = property_type_price.max() - property_type_price.min()
summary_stats['Feature'].append('Property Type')
summary_stats['Impact'].append(f'₹{pt_range:.2f}L difference')
summary_stats['Observation'].append(f'{property_type_price.idxmax()} is most expensive')

# BHK
bhk_increase = bhk_price.iloc[-1] - bhk_price.iloc[0]
summary_stats['Feature'].append('BHK')
summary_stats['Impact'].append(f'₹{bhk_increase:.2f}L increase (1→5 BHK)')
summary_stats['Observation'].append('Linear positive relationship')

# Parking
parking_diff = abs(parking_price['Yes'] - parking_price['No'])
summary_stats['Feature'].append('Parking Space')
summary_stats['Impact'].append(f'₹{parking_diff:.2f}L difference')
summary_stats['Observation'].append(f'{"With" if parking_price["Yes"] > parking_price["No"] else "Without"} parking is higher')

# Security
security_diff = abs(security_price['Yes'] - security_price['No'])
summary_stats['Feature'].append('Security')
summary_stats['Impact'].append(f'₹{security_diff:.2f}L difference')
summary_stats['Observation'].append(f'{"With" if security_price["Yes"] > security_price["No"] else "Without"} security is higher')

# Correlation with price
summary_stats['Feature'].append('Size (Sq Ft)')
summary_stats['Impact'].append(f'Correlation: {correlation:.4f}')
summary_stats['Observation'].append('Strong positive correlation')

summary_df = pd.DataFrame(summary_stats)
print("\n" + summary_df.to_string(index=False))

# Save summary to file
summary_df.to_csv('outputs/eda_summary.csv', index=False)
print("\n✓ Saved: outputs/eda_summary.csv")

print("\n" + "="*80)
print("EDA COMPLETE - All visualizations saved in 'outputs/' directory")
print("="*80)
print("\nGenerated Files:")
print("  1. outputs/01_price_size_distribution.png")
print("  2. outputs/02_price_by_location.png")
print("  3. outputs/03_price_by_characteristics.png")
print("  4. outputs/04_size_vs_price.png")
print("  5. outputs/05_correlation_heatmap.png")
print("  6. outputs/06_amenities_infrastructure_impact.png")
print("  7. outputs/07_owner_facing_analysis.png")
print("  8. outputs/08_top_amenities_analysis.png")
print("  9. outputs/eda_summary.csv")

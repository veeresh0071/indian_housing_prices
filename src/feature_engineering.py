"""
Feature Engineering Module for Real Estate Investment Advisor
Creates investment metrics, appreciation rates, and advanced features for ML models
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(filepath='data/processed_data.csv'):
    """Load the preprocessed dataset"""
    print("Loading processed data...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} features")
    return df


def create_investment_metrics(df):
    """Create investment-related metrics"""
    print("\n" + "-"*60)
    print("Creating Investment Metrics...")
    print("-"*60)
    
    # 1. Value Score: Combination of price efficiency and amenities
    df['Value_Score'] = (
        (1 - df['Price_to_City_Ratio'].clip(0, 2) / 2) * 0.4 +  # Lower price = higher score
        df['Infrastructure_Score'] * 0.3 +
        (df['Amenities_Count'] / df['Amenities_Count'].max()) * 0.3
    )
    print(f"  - Value_Score: min={df['Value_Score'].min():.3f}, max={df['Value_Score'].max():.3f}")
    
    # 2. Price Efficiency: Price per sqft relative to city average
    city_avg_price_sqft = df.groupby('City')['Price_per_SqFt'].transform('mean')
    df['Price_Efficiency'] = city_avg_price_sqft / df['Price_per_SqFt'].replace(0, np.nan)
    df['Price_Efficiency'] = df['Price_Efficiency'].fillna(1).clip(0, 5)
    print(f"  - Price_Efficiency: mean={df['Price_Efficiency'].mean():.3f}")
    
    # 3. Size Value Ratio: Size relative to price
    df['Size_Value_Ratio'] = df['Size_in_SqFt'] / df['Price_in_Lakhs']
    print(f"  - Size_Value_Ratio: mean={df['Size_Value_Ratio'].mean():.2f} sqft/lakh")
    
    # 4. BHK Value: Price per BHK
    df['Price_per_BHK'] = df['Price_in_Lakhs'] / df['BHK']
    print(f"  - Price_per_BHK: mean=₹{df['Price_per_BHK'].mean():.2f}L")
    
    # 5. Investment Potential Score (0-100)
    df['Investment_Potential'] = (
        df['Value_Score'] * 40 +
        df['Price_Efficiency'].clip(0, 2) / 2 * 30 +
        df['Infrastructure_Score'] * 30
    )
    print(f"  - Investment_Potential: mean={df['Investment_Potential'].mean():.1f}/100")
    
    return df


def calculate_appreciation_rates(df):
    """Calculate estimated appreciation rates based on property characteristics"""
    print("\n" + "-"*60)
    print("Calculating Appreciation Rates...")
    print("-"*60)
    
    # Base appreciation rate (historical average ~5-8% for Indian real estate)
    base_rate = 0.06
    
    # Factors affecting appreciation
    # 1. Location factor (cities with higher prices tend to appreciate more)
    city_price_rank = df.groupby('City')['Price_in_Lakhs'].transform('mean').rank(pct=True)
    location_factor = 0.02 * city_price_rank  # 0-2% boost based on city ranking
    
    # 2. Property type factor
    property_type_rates = {
        'Apartment': 0.01,      # Apartments appreciate slightly more
        'Villa': 0.005,
        'Independent House': 0.0
    }
    df['Property_Type_Factor'] = df['Property_Type'].map(property_type_rates).fillna(0)
    
    # 3. Age factor (newer properties appreciate more)
    age_factor = np.where(df['Age_of_Property'] <= 5, 0.015,
                 np.where(df['Age_of_Property'] <= 10, 0.01,
                 np.where(df['Age_of_Property'] <= 20, 0.005, 0)))
    
    # 4. Infrastructure factor
    infra_factor = df['Infrastructure_Score'] * 0.02
    
    # 5. Availability factor (under construction may appreciate more)
    availability_factor = np.where(df['Availability_Status'] == 'Under_Construction', 0.01, 0)
    
    # Calculate annual appreciation rate
    df['Annual_Appreciation_Rate'] = (
        base_rate + 
        location_factor + 
        df['Property_Type_Factor'] + 
        age_factor + 
        infra_factor + 
        availability_factor
    )
    
    # Clip to realistic range (3% - 12%)
    df['Annual_Appreciation_Rate'] = df['Annual_Appreciation_Rate'].clip(0.03, 0.12)
    
    print(f"  - Annual Appreciation Rate: {df['Annual_Appreciation_Rate'].min()*100:.1f}% - {df['Annual_Appreciation_Rate'].max()*100:.1f}%")
    print(f"  - Mean Rate: {df['Annual_Appreciation_Rate'].mean()*100:.2f}%")
    
    return df


def predict_future_prices(df, years=5):
    """Calculate predicted prices for future years"""
    print("\n" + "-"*60)
    print(f"Predicting {years}-Year Future Prices...")
    print("-"*60)
    
    # Compound appreciation formula: Future_Price = Current_Price * (1 + rate)^years
    df['Predicted_Price_5Y'] = df['Price_in_Lakhs'] * (1 + df['Annual_Appreciation_Rate']) ** years
    
    # Calculate absolute and percentage gain
    df['Price_Gain_5Y'] = df['Predicted_Price_5Y'] - df['Price_in_Lakhs']
    df['Price_Gain_Pct_5Y'] = (df['Price_Gain_5Y'] / df['Price_in_Lakhs']) * 100
    
    print(f"  - Current Avg Price: ₹{df['Price_in_Lakhs'].mean():.2f}L")
    print(f"  - Predicted Avg Price (5Y): ₹{df['Predicted_Price_5Y'].mean():.2f}L")
    print(f"  - Avg Price Gain: ₹{df['Price_Gain_5Y'].mean():.2f}L ({df['Price_Gain_Pct_5Y'].mean():.1f}%)")
    
    return df


def create_property_age_features(df):
    """Create property age-related features"""
    print("\n" + "-"*60)
    print("Creating Property Age Features...")
    print("-"*60)
    
    # Age categories
    df['Age_Category'] = pd.cut(
        df['Age_of_Property'],
        bins=[0, 5, 10, 20, 100],
        labels=['New (0-5)', 'Recent (5-10)', 'Moderate (10-20)', 'Old (20+)']
    )
    
    # Remaining useful life estimate (assuming 50-year lifespan)
    df['Remaining_Life_Years'] = np.maximum(50 - df['Age_of_Property'], 0)
    
    # Depreciation factor (older = more depreciated)
    df['Depreciation_Factor'] = 1 - (df['Age_of_Property'] / 50).clip(0, 0.5)
    
    print(f"  - Age Categories: {df['Age_Category'].value_counts().to_dict()}")
    print(f"  - Avg Remaining Life: {df['Remaining_Life_Years'].mean():.1f} years")
    
    return df


def create_floor_features(df):
    """Create floor-related features"""
    print("\n" + "-"*60)
    print("Creating Floor Features...")
    print("-"*60)
    
    # Floor position ratio (0 = ground, 1 = top)
    df['Floor_Position_Ratio'] = df['Floor_No'] / df['Total_Floors'].replace(0, 1)
    
    # Floor categories
    df['Floor_Category'] = np.where(df['Floor_No'] == 0, 'Ground',
                           np.where(df['Floor_No'] <= 3, 'Low',
                           np.where(df['Floor_No'] <= 10, 'Mid',
                           np.where(df['Floor_No'] <= 20, 'High', 'Penthouse'))))
    
    # Is top floor
    df['Is_Top_Floor'] = (df['Floor_No'] == df['Total_Floors']).astype(int)
    
    # Is ground floor
    df['Is_Ground_Floor'] = (df['Floor_No'] == 0).astype(int)
    
    print(f"  - Floor Categories: {df['Floor_Category'].value_counts().to_dict()}")
    print(f"  - Top Floor Properties: {df['Is_Top_Floor'].sum():,}")
    print(f"  - Ground Floor Properties: {df['Is_Ground_Floor'].sum():,}")
    
    return df


def create_location_premium_features(df):
    """Create location premium and ranking features"""
    print("\n" + "-"*60)
    print("Creating Location Premium Features...")
    print("-"*60)
    
    # City price tier (1-5, 5 being most expensive)
    df['City_Price_Tier'] = pd.qcut(
        df['City_Median_Price'], 
        q=5, 
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    # State price tier
    df['State_Price_Tier'] = pd.qcut(
        df['State_Median_Price'], 
        q=5, 
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    # Location premium (how much above/below average)
    overall_median = df['Price_in_Lakhs'].median()
    df['Location_Premium'] = (df['City_Median_Price'] - overall_median) / overall_median
    
    # City property count (market size indicator)
    df['City_Property_Count'] = df.groupby('City')['ID'].transform('count')
    df['City_Market_Size'] = pd.qcut(
        df['City_Property_Count'], 
        q=3, 
        labels=['Small', 'Medium', 'Large'],
        duplicates='drop'
    )
    
    print(f"  - City Price Tiers: 1 (cheapest) to 5 (most expensive)")
    print(f"  - Location Premium Range: {df['Location_Premium'].min():.2%} to {df['Location_Premium'].max():.2%}")
    
    return df


def create_composite_scores(df):
    """Create final composite scores for investment decisions"""
    print("\n" + "-"*60)
    print("Creating Composite Scores...")
    print("-"*60)
    
    # 1. Overall Investment Score (0-100)
    df['Overall_Investment_Score'] = (
        df['Investment_Potential'] * 0.3 +
        df['Price_Gain_Pct_5Y'] * 0.3 +
        df['Value_Score'] * 100 * 0.2 +
        df['Depreciation_Factor'] * 100 * 0.2
    ).clip(0, 100)
    
    # 2. Risk Score (lower = safer investment)
    df['Risk_Score'] = (
        (1 - df['Infrastructure_Score']) * 30 +
        (df['Age_of_Property'] / 35) * 30 +
        (1 - df['Depreciation_Factor']) * 20 +
        (df['Price_to_City_Ratio'].clip(0, 2) / 2) * 20
    ).clip(0, 100)
    
    # 3. Investment Grade (A, B, C, D)
    df['Investment_Grade'] = pd.cut(
        df['Overall_Investment_Score'],
        bins=[0, 40, 55, 70, 100],
        labels=['D', 'C', 'B', 'A']
    )
    
    print(f"  - Overall Investment Score: mean={df['Overall_Investment_Score'].mean():.1f}")
    print(f"  - Risk Score: mean={df['Risk_Score'].mean():.1f}")
    print(f"  - Investment Grades: {df['Investment_Grade'].value_counts().to_dict()}")
    
    return df


def run_feature_engineering_pipeline(input_path='data/processed_data.csv', save_output=True):
    """Run the complete feature engineering pipeline"""
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load processed data
    df = load_processed_data(input_path)
    
    # Create investment metrics
    df = create_investment_metrics(df)
    
    # Calculate appreciation rates
    df = calculate_appreciation_rates(df)
    
    # Predict future prices
    df = predict_future_prices(df, years=5)
    
    # Create property age features
    df = create_property_age_features(df)
    
    # Create floor features
    df = create_floor_features(df)
    
    # Create location premium features
    df = create_location_premium_features(df)
    
    # Create composite scores
    df = create_composite_scores(df)
    
    # Save engineered data
    if save_output:
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/engineered_data.csv', index=False)
        print(f"\n✓ Saved engineered data to 'data/engineered_data.csv'")
        
        # Save feature summary
        new_features = [
            'Value_Score', 'Price_Efficiency', 'Size_Value_Ratio', 'Price_per_BHK',
            'Investment_Potential', 'Annual_Appreciation_Rate', 'Predicted_Price_5Y',
            'Price_Gain_5Y', 'Price_Gain_Pct_5Y', 'Age_Category', 'Remaining_Life_Years',
            'Depreciation_Factor', 'Floor_Position_Ratio', 'Floor_Category',
            'Is_Top_Floor', 'Is_Ground_Floor', 'City_Price_Tier', 'State_Price_Tier',
            'Location_Premium', 'City_Property_Count', 'City_Market_Size',
            'Overall_Investment_Score', 'Risk_Score', 'Investment_Grade'
        ]
        
        with open('data/engineered_features.txt', 'w') as f:
            f.write("ENGINEERED FEATURES:\n")
            f.write("="*40 + "\n\n")
            for feat in new_features:
                if feat in df.columns:
                    f.write(f"- {feat}\n")
        print(f"✓ Saved feature list to 'data/engineered_features.txt'")
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"\nDataset shape: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    df = run_feature_engineering_pipeline()
    
    # Show sample of key engineered features
    print("\nSample of key engineered features:")
    sample_cols = [
        'Price_in_Lakhs', 'Predicted_Price_5Y', 'Price_Gain_Pct_5Y',
        'Overall_Investment_Score', 'Investment_Grade', 'Risk_Score'
    ]
    print(df[sample_cols].head(10))
    
    # Show investment grade distribution
    print("\nInvestment Grade Distribution:")
    print(df['Investment_Grade'].value_counts().sort_index())

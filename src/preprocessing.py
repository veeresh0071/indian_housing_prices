"""
Data Preprocessing Module for Real Estate Investment Advisor
Handles: Categorical encoding, feature scaling, and Good Investment label creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='india_housing_prices.csv'):
    """Load the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} features")
    return df


def create_good_investment_label(df):
    """
    Create 'Good_Investment' binary label based on:
    1. Price below median for the city
    2. Price per sqft below median for the city
    3. Has security
    4. Has parking
    """
    print("\n" + "-"*60)
    print("Creating 'Good Investment' Label...")
    print("-"*60)
    
    # Calculate city-level median prices
    city_median_price = df.groupby('City')['Price_in_Lakhs'].transform('median')
    city_median_price_sqft = df.groupby('City')['Price_per_SqFt'].transform('median')
    
    # Create relative price metrics
    df['Price_vs_City_Median'] = df['Price_in_Lakhs'] / city_median_price
    df['PriceSqFt_vs_City_Median'] = df['Price_per_SqFt'] / city_median_price_sqft
    
    # Good Investment criteria:
    # - Price below or at city median (ratio <= 1.0)
    # - Price per sqft below or at city median (ratio <= 1.0)
    # - Has security OR has parking (at least one amenity)
    
    price_condition = df['Price_vs_City_Median'] <= 1.0
    price_sqft_condition = df['PriceSqFt_vs_City_Median'] <= 1.0
    amenity_condition = (df['Security'] == 'Yes') | (df['Parking_Space'] == 'Yes')
    
    # Good investment if price is good AND has amenities
    df['Good_Investment'] = ((price_condition & price_sqft_condition) | amenity_condition).astype(int)
    
    # Print distribution
    good_count = df['Good_Investment'].sum()
    total = len(df)
    print(f"✓ Good Investment: {good_count:,} ({good_count/total*100:.1f}%)")
    print(f"✓ Not Good Investment: {total-good_count:,} ({(total-good_count)/total*100:.1f}%)")
    
    return df


def parse_amenities(df):
    """Parse amenities column and create binary features for common amenities"""
    print("\n" + "-"*60)
    print("Parsing Amenities...")
    print("-"*60)
    
    # Common amenities to extract
    common_amenities = ['Pool', 'Gym', 'Garden', 'Clubhouse', 'Playground']
    
    for amenity in common_amenities:
        df[f'Has_{amenity}'] = df['Amenities'].str.contains(amenity, case=False, na=False).astype(int)
        count = df[f'Has_{amenity}'].sum()
        print(f"  - Has_{amenity}: {count:,} properties ({count/len(df)*100:.1f}%)")
    
    # Count total amenities
    df['Amenities_Count'] = df['Amenities'].str.count(',') + 1
    df.loc[df['Amenities'].isna() | (df['Amenities'] == ''), 'Amenities_Count'] = 0
    
    print(f"✓ Created {len(common_amenities)} amenity features + Amenities_Count")
    return df


def encode_categorical_features(df):
    """Encode categorical features using Label Encoding"""
    print("\n" + "-"*60)
    print("Encoding Categorical Features...")
    print("-"*60)
    
    # Categorical columns to encode
    categorical_cols = [
        'State', 'City', 'Locality', 'Property_Type', 'Furnished_Status',
        'Facing', 'Public_Transport_Accessibility', 'Parking_Space',
        'Security', 'Owner_Type', 'Availability_Status'
    ]
    
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  - {col}: {len(le.classes_)} unique values encoded")
    
    print(f"✓ Encoded {len(encoders)} categorical features")
    return df, encoders


def scale_numerical_features(df, scaler_type='standard'):
    """
    Scale numerical features
    scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
    """
    print("\n" + "-"*60)
    print(f"Scaling Numerical Features ({scaler_type})...")
    print("-"*60)
    
    # Numerical columns to scale (excluding ID and target)
    numerical_cols = [
        'BHK', 'Size_in_SqFt', 'Price_per_SqFt', 'Year_Built',
        'Floor_No', 'Total_Floors', 'Age_of_Property',
        'Nearby_Schools', 'Nearby_Hospitals', 'Amenities_Count'
    ]
    
    # Filter to existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Create scaled versions
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    for i, col in enumerate(numerical_cols):
        df[f'{col}_Scaled'] = scaled_data[:, i]
        print(f"  - {col}: mean={df[col].mean():.2f} → scaled")
    
    print(f"✓ Scaled {len(numerical_cols)} numerical features")
    return df, scaler, numerical_cols


def create_location_features(df):
    """Create location-based aggregate features"""
    print("\n" + "-"*60)
    print("Creating Location-Based Features...")
    print("-"*60)
    
    # City-level statistics
    df['City_Median_Price'] = df.groupby('City')['Price_in_Lakhs'].transform('median')
    df['City_Avg_Price'] = df.groupby('City')['Price_in_Lakhs'].transform('mean')
    df['City_Price_Std'] = df.groupby('City')['Price_in_Lakhs'].transform('std')
    
    # State-level statistics
    df['State_Median_Price'] = df.groupby('State')['Price_in_Lakhs'].transform('median')
    df['State_Avg_Price'] = df.groupby('State')['Price_in_Lakhs'].transform('mean')
    
    # Price relative to location
    df['Price_to_City_Ratio'] = df['Price_in_Lakhs'] / df['City_Median_Price']
    df['Price_to_State_Ratio'] = df['Price_in_Lakhs'] / df['State_Median_Price']
    
    print("✓ Created city-level features: Median, Avg, Std prices")
    print("✓ Created state-level features: Median, Avg prices")
    print("✓ Created relative price ratios")
    
    return df


def create_infrastructure_score(df):
    """Create composite infrastructure score"""
    print("\n" + "-"*60)
    print("Creating Infrastructure Score...")
    print("-"*60)
    
    # Binary features to numeric
    df['Parking_Binary'] = (df['Parking_Space'] == 'Yes').astype(int)
    df['Security_Binary'] = (df['Security'] == 'Yes').astype(int)
    
    # Transport accessibility score
    transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['Transport_Score'] = df['Public_Transport_Accessibility'].map(transport_map)
    
    # Normalize schools and hospitals (0-1 scale)
    df['Schools_Norm'] = df['Nearby_Schools'] / df['Nearby_Schools'].max()
    df['Hospitals_Norm'] = df['Nearby_Hospitals'] / df['Nearby_Hospitals'].max()
    
    # Composite infrastructure score (weighted average)
    df['Infrastructure_Score'] = (
        df['Parking_Binary'] * 0.15 +
        df['Security_Binary'] * 0.20 +
        df['Transport_Score'] / 3 * 0.25 +
        df['Schools_Norm'] * 0.20 +
        df['Hospitals_Norm'] * 0.20
    )
    
    print(f"✓ Infrastructure Score: min={df['Infrastructure_Score'].min():.2f}, max={df['Infrastructure_Score'].max():.2f}")
    
    return df


def prepare_ml_features(df):
    """Prepare feature sets for ML models"""
    print("\n" + "-"*60)
    print("Preparing ML Feature Sets...")
    print("-"*60)
    
    # Features for classification (Good Investment prediction)
    classification_features = [
        # Encoded categorical
        'Property_Type_Encoded', 'Furnished_Status_Encoded', 'Facing_Encoded',
        'Public_Transport_Accessibility_Encoded', 'Parking_Space_Encoded',
        'Security_Encoded', 'Owner_Type_Encoded', 'Availability_Status_Encoded',
        'City_Encoded', 'State_Encoded',
        # Scaled numerical
        'BHK_Scaled', 'Size_in_SqFt_Scaled', 'Age_of_Property_Scaled',
        'Floor_No_Scaled', 'Total_Floors_Scaled',
        'Nearby_Schools_Scaled', 'Nearby_Hospitals_Scaled',
        # Amenity features
        'Has_Pool', 'Has_Gym', 'Has_Garden', 'Has_Clubhouse', 'Has_Playground',
        'Amenities_Count',
        # Infrastructure
        'Infrastructure_Score'
    ]
    
    # Features for regression (Price prediction)
    regression_features = classification_features + [
        'City_Median_Price', 'State_Median_Price',
        'Price_to_City_Ratio', 'Price_to_State_Ratio'
    ]
    
    # Filter to existing columns
    classification_features = [f for f in classification_features if f in df.columns]
    regression_features = [f for f in regression_features if f in df.columns]
    
    print(f"✓ Classification features: {len(classification_features)}")
    print(f"✓ Regression features: {len(regression_features)}")
    
    return classification_features, regression_features


def split_data(df, target_col, feature_cols, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets"""
    print("\n" + "-"*60)
    print("Splitting Data...")
    print("-"*60)
    
    X = df[feature_cols]
    y = df[target_col]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() < 10 else None
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp if y_temp.nunique() < 10 else None
    )
    
    print(f"✓ Training set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"✓ Validation set: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"✓ Test set: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing_pipeline(filepath='india_housing_prices.csv', save_output=True):
    """Run the complete preprocessing pipeline"""
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load data
    df = load_data(filepath)
    
    # 2. Parse amenities
    df = parse_amenities(df)
    
    # 3. Create location features
    df = create_location_features(df)
    
    # 4. Create infrastructure score
    df = create_infrastructure_score(df)
    
    # 5. Create Good Investment label
    df = create_good_investment_label(df)
    
    # 6. Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # 7. Scale numerical features
    df, scaler, scaled_cols = scale_numerical_features(df, scaler_type='standard')
    
    # 8. Prepare ML feature sets
    clf_features, reg_features = prepare_ml_features(df)
    
    # Save processed data
    if save_output:
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/processed_data.csv', index=False)
        print(f"\n✓ Saved processed data to 'data/processed_data.csv'")
        
        # Save feature lists
        with open('data/feature_lists.txt', 'w') as f:
            f.write("CLASSIFICATION FEATURES:\n")
            f.write('\n'.join(clf_features))
            f.write("\n\nREGRESSION FEATURES:\n")
            f.write('\n'.join(reg_features))
        print(f"✓ Saved feature lists to 'data/feature_lists.txt'")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nDataset shape: {df.shape}")
    print(f"New features created: {len(df.columns) - 23}")
    
    return df, encoders, scaler, clf_features, reg_features


if __name__ == "__main__":
    df, encoders, scaler, clf_features, reg_features = run_preprocessing_pipeline()
    
    # Show sample of processed data
    print("\nSample of key processed features:")
    sample_cols = ['Price_in_Lakhs', 'Good_Investment', 'Infrastructure_Score', 
                   'Price_to_City_Ratio', 'Property_Type_Encoded']
    print(df[sample_cols].head(10))

# Real Estate Investment Advisor

ML application to predict property profitability and future values for real estate investors.

## Dataset Overview
- **Size**: 250,000 properties
- **Features**: 23 columns
- **Location Coverage**: 20 states, 42 cities, 500 localities
- **Data Quality**: No missing values, no duplicates

## Key Statistics
- **Price Range**: ₹10L - ₹500L (Avg: ₹254.59L)
- **Property Size**: 500 - 5,000 sqft (Avg: 2,750 sqft)
- **BHK**: 1-5 bedrooms (Avg: 3)
- **Property Age**: 2-35 years (Avg: 18 years)
- **Property Types**: Villa (33.5%), Independent House (33.3%), Apartment (33.2%)

## Project Structure
```
p4/
├── data/                      # Processed datasets
├── notebooks/                 # Jupyter notebooks for EDA
├── src/                       # Source code
│   ├── preprocessing.py       # Data preprocessing
│   ├── eda.py                # Exploratory analysis
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py     # ML model training
│   └── app.py                # Streamlit application
├── models/                    # Saved ML models
├── outputs/                   # Plots and reports
├── india_housing_prices.csv  # Original dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Tasks Checklist

### 1. Data Preprocessing ✓ COMPLETE
- [x] Handle encoding of categorical features (11 features label-encoded)
- [x] Feature scaling/normalization (10 numerical features standardized)
- [x] Create "Good Investment" label (85% good, 15% not good)
- [x] Parse amenities into binary features (5 amenity flags + count)
- [x] Create location-based features (city/state median prices, ratios)
- [x] Create infrastructure score (composite metric)

### 2. EDA
- [ ] Price distribution analysis
- [ ] Location-wise analysis
- [ ] Correlation analysis
- [ ] Impact of amenities on price
- [ ] Visualization creation

### 3. Feature Engineering ✓ COMPLETE
- [x] Create investment metrics (Value_Score, Price_Efficiency, Investment_Potential)
- [x] Engineer location-based features (City/State tiers, Location_Premium, Market_Size)
- [x] Calculate appreciation rates (3-12% annual, 5-year price predictions)
- [x] Property age features (Age_Category, Remaining_Life, Depreciation_Factor)
- [x] Floor features (Floor_Category, Is_Top_Floor, Is_Ground_Floor)
- [x] Composite scores (Overall_Investment_Score, Risk_Score, Investment_Grade)

### 4. Model Development ✓ COMPLETE
- [x] Classification: Good Investment predictor (XGBoost - F1: 0.9998, ROC-AUC: 1.0)
- [x] Regression: 5-year price prediction (XGBoost - RMSE: ₹2.71L, R²: 0.9999)
- [x] Model evaluation and comparison (4 classifiers, 5 regressors tested)
- [x] Models saved to models/ directory

### 5. Streamlit App ✓ COMPLETE
- [x] User input form (property details, location, amenities, infrastructure)
- [x] Filtering functionality (multi-criteria property search with export)
- [x] Prediction display (investment recommendation + 5-year price forecast)
- [x] Visualization dashboard (market trends, price distribution, investment grades)

## Technologies
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **ML**: Scikit-learn, XGBoost
- **Deployment**: Streamlit
- **Experiment Tracking**: MLflow

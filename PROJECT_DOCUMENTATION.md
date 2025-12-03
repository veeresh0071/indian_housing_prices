# Real Estate Investment Advisor - Project Documentation

## Executive Summary

This project develops a machine learning application to assist potential investors in making real estate decisions. The system classifies whether a property is a "Good Investment" and predicts the estimated property price after 5 years using a dataset of 250,000 Indian properties.

---

## 1. Problem Statement

### Objectives
1. **Classification**: Predict whether a property is a "Good Investment" (Yes/No)
2. **Regression**: Predict the estimated property price after 5 years
3. **Deployment**: User-interactive Streamlit application with investment recommendations
4. **Tracking**: MLflow integration for experiment tracking and model versioning

### Business Use Cases
- Empower real estate investors with intelligent tools to assess long-term returns
- Support buyers in choosing high-return properties in developing areas
- Help real estate companies automate investment analysis for listings
- Improve customer trust in real estate platforms with data-backed predictions

---

## 2. Dataset Overview

### Source
- **File**: `india_housing_prices.csv`
- **Records**: 250,000 properties
- **Features**: 23 columns
- **Coverage**: 20 states, 42 cities, 500 localities

### Feature Categories

| Category | Features |
|----------|----------|
| Location | State, City, Locality |
| Property | Property_Type, BHK, Size_in_SqFt, Facing, Furnished_Status |
| Building | Floor_No, Total_Floors, Year_Built, Age_of_Property |
| Pricing | Price_in_Lakhs, Price_per_SqFt |
| Infrastructure | Nearby_Schools, Nearby_Hospitals, Public_Transport_Accessibility |
| Amenities | Parking_Space, Security, Amenities |
| Ownership | Owner_Type, Availability_Status |

### Data Quality
- **Missing Values**: None
- **Duplicates**: None
- **Data Types**: Properly formatted

---

## 3. Methodology

### 3.1 Data Preprocessing (`src/preprocessing.py`)

**Steps Performed:**
1. **Categorical Encoding**: Label encoding for 11 categorical features
2. **Feature Scaling**: StandardScaler for 10 numerical features
3. **Amenities Parsing**: Extracted 5 binary amenity flags (Pool, Gym, Garden, Clubhouse, Playground)
4. **Location Features**: City/State median prices, price ratios
5. **Infrastructure Score**: Composite metric combining parking, security, transport, schools, hospitals
6. **Target Creation**: "Good Investment" binary label based on:
   - Price vs city median
   - Price per sqft vs city median
   - Presence of security/parking

**Output**: `data/processed_data.csv` (66 features)

### 3.2 Exploratory Data Analysis (`src/eda.py`)

**Key Findings:**

1. **Price Distribution**
   - Range: ₹10L - ₹500L
   - Mean: ₹254.59L, Median: ₹253.87L
   - Nearly symmetric distribution

2. **Location Analysis**
   - Top states by price: Karnataka (₹257L), Tamil Nadu (₹257L), West Bengal (₹256L)
   - Relatively uniform pricing across major cities

3. **Property Characteristics**
   - Independent House > Villa > Apartment (by avg price)
   - Weak correlation between size and price (-0.0025)

4. **Infrastructure Impact**
   - Security: +₹1.08L price premium
   - Parking: +₹0.32L price premium

5. **Correlation Analysis**
   - Price_per_SqFt: 0.5556 (strongest predictor)
   - Most numerical features show negligible correlation with price
   - Categorical features (location, amenities) are more important

**Visualizations Generated**: 8 plots in `outputs/` folder

### 3.3 Feature Engineering (`src/feature_engineering.py`)

**Features Created:**

| Feature | Description |
|---------|-------------|
| Value_Score | Combination of price efficiency and amenities (0-1) |
| Price_Efficiency | Price per sqft vs city average |
| Size_Value_Ratio | Sqft per lakh (avg: 22 sqft/lakh) |
| Investment_Potential | 0-100 investment score |
| Annual_Appreciation_Rate | 6.3% - 12% based on property factors |
| Predicted_Price_5Y | 5-year projected price |
| Age_Category | New/Recent/Moderate/Old |
| Floor_Category | Ground/Low/Mid/High/Penthouse |
| City_Price_Tier | 1-5 ranking by city prices |
| Infrastructure_Score | Composite infrastructure metric |
| Overall_Investment_Score | Final 0-100 score |
| Risk_Score | Investment risk metric |
| Investment_Grade | A/B/C/D classification |

**Output**: `data/engineered_data.csv` (91 features)

### 3.4 Model Development (`src/model_training.py`)

#### Classification Models (Good Investment Prediction)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 98.56% | 99.14% | 99.17% | 0.9916 | 0.9989 |
| Random Forest | 99.90% | 99.97% | 99.91% | 0.9994 | 1.0000 |
| Gradient Boosting | 99.93% | 99.97% | 99.95% | 0.9996 | 1.0000 |
| **XGBoost** ✓ | **99.96%** | **99.98%** | **99.98%** | **0.9998** | **1.0000** |

#### Regression Models (5-Year Price Prediction)

| Model | RMSE (₹L) | MAE (₹L) | R² | MAPE |
|-------|-----------|----------|-----|------|
| Linear Regression | 13.91 | 10.67 | 0.9962 | 5.40% |
| Ridge Regression | 13.91 | 10.67 | 0.9962 | 5.40% |
| Random Forest | 10.18 | 7.39 | 0.9979 | 2.00% |
| Gradient Boosting | 2.89 | 2.11 | 0.9998 | 0.76% |
| **XGBoost** ✓ | **2.71** | **2.10** | **0.9999** | **0.83%** |

#### Feature Importance

**Classification (Top 5):**
1. Parking_Space_Encoded (38.7%)
2. Security_Encoded (26.7%)
3. Infrastructure_Score (17.8%)
4. Price_per_SqFt (12.3%)
5. Price_in_Lakhs (4.2%)

**Regression (Top 5):**
1. Price_in_Lakhs (95.5%)
2. Availability_Status_Encoded (1.3%)
3. Age_of_Property (0.9%)
4. City_Median_Price (0.8%)
5. Property_Type_Encoded (0.7%)

---

## 4. Streamlit Application (`src/app.py`)

### Features

#### 🔮 Investment Predictor
- Property input form (location, size, BHK, amenities, etc.)
- Real-time investment classification with confidence score
- 5-year price prediction with gain calculation
- Interactive price projection chart
- Feature importance visualization

#### 📊 Market Dashboard
- Filter by state, city, property type
- Key metrics display (total properties, avg price, investment %)
- Price distribution histogram
- Investment grade pie chart
- Price vs size scatter plot
- Location heatmaps (treemap + matrix)
- Top cities analysis table

#### 🔍 Property Filter
- Multi-criteria filtering (state, type, price, size, BHK, grade)
- Good investments only toggle
- Sortable results
- CSV export functionality

### Running the Application
```bash
streamlit run src/app.py --server.port 8502
```

---

## 5. MLflow Integration

### Experiment Tracking
- **Experiment Name**: Real_Estate_Investment_Advisor
- **Tracking URI**: `./mlruns`
- **Runs Logged**: 9 (4 classification + 5 regression)

### Registered Models
1. **GoodInvestment_Classifier** (XGBoost) - Version 1
2. **Price_5Y_Predictor** (XGBoost) - Version 1

### Logged Information
- Parameters: model_name, model_type, n_features, train/test samples
- Metrics: All evaluation metrics per model
- Artifacts: Trained model files with signatures

### Viewing MLflow UI
```bash
mlflow ui --port 5000
```
Open http://localhost:5000

---

## 6. Project Structure

```
p4/
├── data/
│   ├── processed_data.csv      # Preprocessed dataset
│   ├── engineered_data.csv     # Feature-engineered dataset
│   ├── feature_lists.txt       # Feature names
│   └── engineered_features.txt # Engineered feature list
├── models/
│   ├── classification_model.pkl # Best classifier (XGBoost)
│   ├── regression_model.pkl     # Best regressor (XGBoost)
│   ├── model_features.txt       # Model feature lists
│   └── model_info.pkl           # Model metadata
├── mlruns/                      # MLflow tracking data
├── outputs/
│   ├── 01-08_*.png             # EDA visualizations
│   ├── classification_comparison.csv
│   ├── regression_comparison.csv
│   └── eda_summary.csv
├── src/
│   ├── preprocessing.py        # Data preprocessing
│   ├── eda.py                  # Exploratory analysis
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # ML model training
│   └── app.py                  # Streamlit application
├── india_housing_prices.csv    # Original dataset
├── requirements.txt            # Dependencies
├── README.md                   # Project overview
└── PROJECT_DOCUMENTATION.md    # This file
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# 1. Preprocessing
python src/preprocessing.py

# 2. Feature Engineering
python src/feature_engineering.py

# 3. Model Training (with MLflow)
python src/model_training.py

# 4. Launch Application
streamlit run src/app.py
```

---

## 8. Results Summary

### Model Performance
- **Classification**: 99.96% accuracy, 0.9998 F1-score (XGBoost)
- **Regression**: ₹2.71L RMSE, 0.9999 R² (XGBoost)

### Investment Analysis
- **Good Investments**: 85% of properties
- **Investment Grades**: A (12%), B (56%), C (31%), D (1%)
- **Average 5-Year Gain**: 58.2% (₹148L on avg ₹255L property)

### Key Insights
1. Parking and security are the strongest predictors of good investments
2. Location-based features are more important than property size
3. Infrastructure score significantly impacts investment potential
4. Current price is the dominant factor for future price prediction

---

## 9. Technologies Used

| Category | Technologies |
|----------|-------------|
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Deployment | Streamlit |
| Experiment Tracking | MLflow |
| Development | Python 3.10 |

---

## 10. Future Enhancements

1. **Real-time Data**: Integration with property listing APIs
2. **Geographic Visualization**: Interactive maps with property locations
3. **Advanced Models**: Deep learning for price prediction
4. **User Authentication**: Save user preferences and history
5. **API Deployment**: REST API for model predictions
6. **Mobile App**: Cross-platform mobile application

---

## 11. References

- Dataset: India Housing Prices Dataset
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Streamlit Documentation: https://docs.streamlit.io/
- MLflow Documentation: https://mlflow.org/docs/latest/

---

**Project Completed**: December 2024
**Author**: Real Estate Investment Advisor Team

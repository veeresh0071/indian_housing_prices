# 🏠 Real Estate Investment Advisor

AI-powered ML application to predict property profitability and future values for real estate investors in India.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.5+-orange.svg)

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Total Properties | 250,000 |
| Features | 23 columns |
| States | 20 |
| Cities | 42 |
| Localities | 500 |
| Price Range | ₹10L - ₹500L |
| Property Size | 500 - 5,000 sqft |

## 🎯 Model Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| XGBoost Classifier | Good Investment Prediction | F1 Score | 0.9998 |
| XGBoost Classifier | Good Investment Prediction | ROC-AUC | 1.0000 |
| XGBoost Regressor | 5-Year Price Prediction | RMSE | ₹2.71L |
| XGBoost Regressor | 5-Year Price Prediction | R² | 0.9999 |

## 🖥️ Streamlit Application

### Page 1: 🔮 Investment Predictor
Enter property details to get AI-powered investment recommendations and 5-year price forecasts.

**Features:**
- Property input form (location, size, BHK, amenities, infrastructure)
- Real-time investment classification with confidence score
- 5-year price prediction with gain calculation
- Interactive price projection chart
- Feature importance visualization

```
┌─────────────────────────────────────────────────────────────┐
│  🏠 Real Estate Investment Advisor                          │
│  ─────────────────────────────────────────────────────────  │
│  📍 Location          🏗️ Property Details    🏪 Amenities   │
│  ┌──────────────┐    ┌──────────────┐      ☑ Pool          │
│  │ State: [▼]  │    │ Type: [▼]   │      ☑ Gym           │
│  │ City:  [▼]  │    │ BHK:  [3]   │      ☐ Garden        │
│  └──────────────┘    │ Size: [2000]│      ☐ Clubhouse     │
│                      │ Age:  [10]  │                       │
│                      └──────────────┘                       │
│  [🔮 Analyze Investment]                                    │
│  ─────────────────────────────────────────────────────────  │
│  📈 Results:  ✅ GOOD INVESTMENT (95.2% confidence)         │
│  Current: ₹200L → 5-Year: ₹316L (+₹116L, +58%)             │
└─────────────────────────────────────────────────────────────┘
```

### Page 2: 📊 Market Dashboard
Explore real estate market trends and insights with interactive visualizations.

**Features:**
- Filter by state, city, property type
- Key metrics display (total properties, avg price, investment %)
- Price distribution histogram
- Investment grade pie chart
- Location heatmaps (treemap + matrix)
- Top cities analysis table

```
┌─────────────────────────────────────────────────────────────┐
│  📊 Market Dashboard                                        │
│  ─────────────────────────────────────────────────────────  │
│  [State: All ▼] [City: All ▼] [Type: All ▼]                │
│  ─────────────────────────────────────────────────────────  │
│  📈 Key Metrics                                             │
│  ┌────────────┬────────────┬────────────┬────────────┐     │
│  │ Properties │ Avg Price  │ Avg Size   │ Good Inv % │     │
│  │  250,000   │  ₹254.59L  │ 2,750 sqft │   85.0%    │     │
│  └────────────┴────────────┴────────────┴────────────┘     │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ Price Distribution  │  │ Investment Grades   │          │
│  │     📊 Histogram    │  │     🥧 Pie Chart    │          │
│  └─────────────────────┘  └─────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Page 3: 🔍 Property Filter
Search and filter properties matching your investment criteria.

**Features:**
- Multi-criteria filtering (state, type, price, size, BHK, grade)
- Good investments only toggle
- Sortable results by price, score, or size
- CSV export functionality
- Shows top 100 matching properties

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 Property Filter                                         │
│  ─────────────────────────────────────────────────────────  │
│  State: [Multi-select]  Price: [₹50L ──●── ₹300L]          │
│  Type:  [Multi-select]  Size:  [1000 ──●── 3000]           │
│  BHK:   [1,2,3,4,5]     ☑ Good Investments Only            │
│  Grade: [A,B,C,D]       Sort: [Investment Score ▼]         │
│  ─────────────────────────────────────────────────────────  │
│  Found 45,230 properties                                    │
│  ┌──────┬───────┬─────┬────────┬───────┬─────────┬────────┐│
│  │ City │ Type  │ BHK │ Size   │ Price │ Grade   │ 5Y Est ││
│  ├──────┼───────┼─────┼────────┼───────┼─────────┼────────┤│
│  │ Pune │ Villa │  3  │ 2,500  │ ₹180L │   A     │ ₹285L  ││
│  │ ...  │ ...   │ ... │  ...   │  ...  │  ...    │  ...   ││
│  └──────┴───────┴─────┴────────┴───────┴─────────┴────────┘│
│  [📥 Download Results (CSV)]                                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/veeresh0071/indian_housing_prices.git
cd indian_housing_prices

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run src/app.py
```
Open http://localhost:8501 in your browser.

### Run the Pipeline (Optional)
```bash
# 1. Data Preprocessing
python src/preprocessing.py

# 2. Feature Engineering
python src/feature_engineering.py

# 3. Model Training with MLflow
python src/model_training.py

# 4. View MLflow UI
mlflow ui --port 5000
```

## 📁 Project Structure

```
indian_housing_prices/
├── data/
│   ├── processed_data.csv       # Preprocessed dataset
│   ├── engineered_data.csv      # Feature-engineered dataset
│   └── feature_lists.txt        # Feature names
├── models/
│   ├── classification_model.pkl # XGBoost classifier
│   ├── regression_model.pkl     # XGBoost regressor
│   └── model_info.pkl           # Model metadata
├── mlruns/                      # MLflow experiment tracking
├── outputs/
│   ├── *.png                    # EDA visualizations
│   └── *.csv                    # Model comparison reports
├── src/
│   ├── app.py                   # Streamlit application
│   ├── preprocessing.py         # Data preprocessing
│   ├── eda.py                   # Exploratory analysis
│   ├── feature_engineering.py   # Feature creation
│   └── model_training.py        # ML model training
├── india_housing_prices.csv     # Original dataset
├── requirements.txt             # Python dependencies
├── PROJECT_DOCUMENTATION.md     # Detailed documentation
└── README.md                    # This file
```

## 🔧 Technologies Used

| Category | Technologies |
|----------|-------------|
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Deployment | Streamlit |
| Experiment Tracking | MLflow |
| Version Control | Git, Git LFS |

## 📈 Key Features

- **Investment Classification**: Predicts if a property is a good investment (Yes/No)
- **Price Forecasting**: Estimates property value after 5 years
- **Market Analysis**: Interactive dashboards with location-wise insights
- **Property Search**: Filter and export properties matching criteria
- **MLflow Integration**: Full experiment tracking and model versioning

## 📝 Documentation

- [Project Documentation](PROJECT_DOCUMENTATION.md) - Detailed methodology and findings
- [EDA Report](EDA_FINDINGS_REPORT.md) - Exploratory data analysis results
- [Data Summary](DATA_EXPLORATION_SUMMARY.md) - Dataset overview

## 👤 Author

**Veeresh**
- GitHub: [@veeresh0071](https://github.com/veeresh0071)

## 📄 License

This project is for educational purposes.

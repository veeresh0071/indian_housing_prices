"""
Model Training Module for Real Estate Investment Advisor
Trains classification (Good Investment) and regression (Price Prediction) models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import mlflow, but make it optional
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Continuing without experiment tracking.")


def setup_mlflow():
    """Setup MLflow tracking"""
    if not MLFLOW_AVAILABLE:
        return False
    
    # Set tracking URI to local mlruns folder
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set experiment
    experiment_name = "Real_Estate_Investment_Advisor"
    mlflow.set_experiment(experiment_name)
    
    print(f"✓ MLflow tracking enabled")
    print(f"  Experiment: {experiment_name}")
    print(f"  Tracking URI: ./mlruns")
    
    return True


def load_engineered_data(filepath='data/engineered_data.csv'):
    """Load the engineered dataset"""
    print("Loading engineered data...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} features")
    return df


def prepare_classification_data(df):
    """Prepare data for classification model (Good Investment prediction)"""
    print("\n" + "-"*60)
    print("Preparing Classification Data...")
    print("-"*60)
    
    # Features for classification
    feature_cols = [
        'Property_Type_Encoded', 'Furnished_Status_Encoded', 'Facing_Encoded',
        'Public_Transport_Accessibility_Encoded', 'Parking_Space_Encoded',
        'Security_Encoded', 'Owner_Type_Encoded', 'Availability_Status_Encoded',
        'City_Encoded', 'State_Encoded',
        'BHK', 'Size_in_SqFt', 'Age_of_Property', 'Floor_No', 'Total_Floors',
        'Nearby_Schools', 'Nearby_Hospitals',
        'Has_Pool', 'Has_Gym', 'Has_Garden', 'Has_Clubhouse', 'Has_Playground',
        'Amenities_Count', 'Infrastructure_Score',
        'Price_in_Lakhs', 'Price_per_SqFt'
    ]
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Good_Investment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Training set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    print(f"✓ Target distribution - Good: {y.sum():,} ({y.mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, feature_cols


def prepare_regression_data(df):
    """Prepare data for regression model (5-year price prediction)"""
    print("\n" + "-"*60)
    print("Preparing Regression Data...")
    print("-"*60)
    
    # Features for regression (exclude price-related targets)
    feature_cols = [
        'Property_Type_Encoded', 'Furnished_Status_Encoded', 'Facing_Encoded',
        'Public_Transport_Accessibility_Encoded', 'Parking_Space_Encoded',
        'Security_Encoded', 'Owner_Type_Encoded', 'Availability_Status_Encoded',
        'City_Encoded', 'State_Encoded',
        'BHK', 'Size_in_SqFt', 'Age_of_Property', 'Floor_No', 'Total_Floors',
        'Nearby_Schools', 'Nearby_Hospitals',
        'Has_Pool', 'Has_Gym', 'Has_Garden', 'Has_Clubhouse', 'Has_Playground',
        'Amenities_Count', 'Infrastructure_Score',
        'City_Median_Price', 'State_Median_Price',
        'Price_in_Lakhs'  # Current price as input
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Predicted_Price_5Y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Training set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    print(f"✓ Target range: ₹{y.min():.2f}L - ₹{y.max():.2f}L")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate classification models"""
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = (name, model)
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"clf_{name.replace(' ', '_')}", nested=True):
                # Log parameters
                mlflow.log_param("model_type", "classification")
                mlflow.log_param("model_name", name)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                
                # Log metrics
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc
                })
                
                # Log model with signature
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(model, "model", signature=signature)
    
    print(f"\n✓ Best Classification Model: {best_model[0]} (F1: {best_score:.4f})")
    return results, best_model


def train_regression_models(X_train, X_test, y_train, y_test):
    """Train and evaluate regression models"""
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODELS")
    print("="*60)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = float('inf')
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        print(f"  RMSE:  ₹{rmse:.2f}L")
        print(f"  MAE:   ₹{mae:.2f}L")
        print(f"  R²:    {r2:.4f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        # Track best model (lowest RMSE)
        if rmse < best_score:
            best_score = rmse
            best_model = (name, model)
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"reg_{name.replace(' ', '_')}", nested=True):
                # Log parameters
                mlflow.log_param("model_type", "regression")
                mlflow.log_param("model_name", name)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                
                # Log metrics
                mlflow.log_metrics({
                    "rmse": rmse,
                    "mae": mae,
                    "r2_score": r2,
                    "mape": mape
                })
                
                # Log model with signature
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(model, "model", signature=signature)
    
    print(f"\n✓ Best Regression Model: {best_model[0]} (RMSE: ₹{best_score:.2f}L)")
    return results, best_model


def get_feature_importance(model, feature_names, model_type='tree'):
    """Extract feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def save_models(clf_model, reg_model, clf_features, reg_features):
    """Save trained models and feature lists"""
    print("\n" + "-"*60)
    print("Saving Models...")
    print("-"*60)
    
    os.makedirs('models', exist_ok=True)
    
    # Save classification model
    joblib.dump(clf_model[1], 'models/classification_model.pkl')
    print(f"✓ Saved: models/classification_model.pkl ({clf_model[0]})")
    
    # Save regression model
    joblib.dump(reg_model[1], 'models/regression_model.pkl')
    print(f"✓ Saved: models/regression_model.pkl ({reg_model[0]})")
    
    # Save feature lists
    with open('models/model_features.txt', 'w') as f:
        f.write("CLASSIFICATION FEATURES:\n")
        f.write('\n'.join(clf_features))
        f.write("\n\nREGRESSION FEATURES:\n")
        f.write('\n'.join(reg_features))
    print("✓ Saved: models/model_features.txt")
    
    # Save model info
    model_info = {
        'classification_model': clf_model[0],
        'regression_model': reg_model[0],
        'clf_features': clf_features,
        'reg_features': reg_features
    }
    joblib.dump(model_info, 'models/model_info.pkl')
    print("✓ Saved: models/model_info.pkl")


def create_model_comparison_report(clf_results, reg_results):
    """Create a comparison report of all models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)
    
    # Classification comparison
    print("\n--- Classification Models ---")
    clf_df = pd.DataFrame({
        name: {k: v for k, v in metrics.items() if k != 'model'}
        for name, metrics in clf_results.items()
    }).T
    print(clf_df.round(4).to_string())
    
    # Regression comparison
    print("\n--- Regression Models ---")
    reg_df = pd.DataFrame({
        name: {k: v for k, v in metrics.items() if k != 'model'}
        for name, metrics in reg_results.items()
    }).T
    print(reg_df.round(4).to_string())
    
    # Save reports
    os.makedirs('outputs', exist_ok=True)
    clf_df.to_csv('outputs/classification_comparison.csv')
    reg_df.to_csv('outputs/regression_comparison.csv')
    print("\n✓ Saved comparison reports to outputs/")
    
    return clf_df, reg_df


def register_best_models(best_clf, best_reg, clf_features, reg_features, X_train_clf, X_train_reg):
    """Register best models in MLflow Model Registry"""
    if not MLFLOW_AVAILABLE:
        return
    
    print("\n" + "-"*60)
    print("Registering Best Models in MLflow...")
    print("-"*60)
    
    # Register classification model
    with mlflow.start_run(run_name="best_classification_model"):
        mlflow.log_param("model_name", best_clf[0])
        mlflow.log_param("model_type", "classification")
        mlflow.log_param("features", clf_features)
        
        signature = infer_signature(X_train_clf.head(1), np.array([0]))
        mlflow.sklearn.log_model(
            best_clf[1], 
            "classification_model",
            signature=signature,
            registered_model_name="GoodInvestment_Classifier"
        )
        print(f"✓ Registered: GoodInvestment_Classifier ({best_clf[0]})")
    
    # Register regression model
    with mlflow.start_run(run_name="best_regression_model"):
        mlflow.log_param("model_name", best_reg[0])
        mlflow.log_param("model_type", "regression")
        mlflow.log_param("features", reg_features)
        
        signature = infer_signature(X_train_reg.head(1), np.array([0.0]))
        mlflow.sklearn.log_model(
            best_reg[1], 
            "regression_model",
            signature=signature,
            registered_model_name="Price_5Y_Predictor"
        )
        print(f"✓ Registered: Price_5Y_Predictor ({best_reg[0]})")


def run_model_training_pipeline(data_path='data/engineered_data.csv'):
    """Run the complete model training pipeline"""
    print("="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Setup MLflow
    mlflow_enabled = setup_mlflow() if MLFLOW_AVAILABLE else False
    
    # Load data
    df = load_engineered_data(data_path)
    
    # Prepare classification data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, clf_features = prepare_classification_data(df)
    
    # Prepare regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, reg_features = prepare_regression_data(df)
    
    # Start parent MLflow run
    if MLFLOW_AVAILABLE:
        mlflow.start_run(run_name="model_training_pipeline")
    
    try:
        # Train classification models
        clf_results, best_clf = train_classification_models(
            X_train_clf, X_test_clf, y_train_clf, y_test_clf
        )
        
        # Train regression models
        reg_results, best_reg = train_regression_models(
            X_train_reg, X_test_reg, y_train_reg, y_test_reg
        )
    finally:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
    
    # Get feature importance for best models
    print("\n" + "-"*60)
    print("Feature Importance (Top 10)")
    print("-"*60)
    
    clf_importance = get_feature_importance(best_clf[1], clf_features)
    if clf_importance is not None:
        print(f"\n{best_clf[0]} - Classification:")
        print(clf_importance.head(10).to_string(index=False))
    
    reg_importance = get_feature_importance(best_reg[1], reg_features)
    if reg_importance is not None:
        print(f"\n{best_reg[0]} - Regression:")
        print(reg_importance.head(10).to_string(index=False))
    
    # Save models locally
    save_models(best_clf, best_reg, clf_features, reg_features)
    
    # Register best models in MLflow
    if MLFLOW_AVAILABLE:
        register_best_models(best_clf, best_reg, clf_features, reg_features, X_train_clf, X_train_reg)
    
    # Create comparison report
    create_model_comparison_report(clf_results, reg_results)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Classification Model: {best_clf[0]}")
    print(f"Best Regression Model: {best_reg[0]}")
    print("\nSaved files:")
    print("  - models/classification_model.pkl")
    print("  - models/regression_model.pkl")
    print("  - models/model_features.txt")
    print("  - models/model_info.pkl")
    print("  - outputs/classification_comparison.csv")
    print("  - outputs/regression_comparison.csv")
    
    if MLFLOW_AVAILABLE:
        print("\nMLflow:")
        print("  - Experiments logged to ./mlruns")
        print("  - Registered models: GoodInvestment_Classifier, Price_5Y_Predictor")
        print("  - View UI: mlflow ui --port 5000")
    
    return clf_results, reg_results, best_clf, best_reg


if __name__ == "__main__":
    clf_results, reg_results, best_clf, best_reg = run_model_training_pipeline()

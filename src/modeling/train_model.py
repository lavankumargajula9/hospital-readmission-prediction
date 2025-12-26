"""
Model Training Module with MLflow Integration
Trains multiple ML models for hospital readmission prediction with experiment tracking
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json

# MLflow imports
import mlflow
import mlflow.sklearn

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Trains and logs ML models with MLflow"""
    
    def __init__(self, mlflow_uri='http://localhost:5000', experiment_name='Hospital_Readmission_Prediction'):
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models_trained = {}
        
        # Set MLflow URI
        mlflow.set_tracking_uri(self.mlflow_uri)
    
    def load_features(self, features_path='data/processed/features_engineered.csv'):
        """Load engineered features"""
        print(f"Loading features from {features_path}...")
        
        try:
            self.features_df = pd.read_csv(features_path)
            print(f"✓ Loaded {len(self.features_df)} samples with {len(self.features_df.columns)} features")
            return True
        except Exception as e:
            print(f"❌ Failed to load features: {str(e)}")
            return False
    
    def prepare_data(self, test_size=0.2, random_state=42, stratify=True):
        """Prepare data for modeling"""
        print(f"\nPreparing data for modeling...")
        
        # Separate features and target
        # Exclude non-predictive columns
        exclude_cols = ['patient_id', 'admission_id', 'created_at', 'age_group', 
                       'comorbidity_burden', 'medication_burden', 'los_category']
        
        feature_cols = [col for col in self.features_df.columns 
                       if col not in exclude_cols + ['readmitted_30_days']]
        
        X = self.features_df[feature_cols].copy()
        y = self.features_df['readmitted_30_days'].copy()
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Handle categorical variables (one-hot encode)
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"Encoding categorical features: {categorical_cols}")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data
        if stratify:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for easier handling
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=X.columns, index=self.X_train.index)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=X.columns, index=self.X_test.index)
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        print(f"✓ Features scaled using StandardScaler")
        
        return True
    
    def create_experiment(self):
        """Create MLflow experiment"""
        print(f"\nSetting up MLflow experiment: {self.experiment_name}")
        
        try:
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"✓ Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"✓ Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(self.experiment_name)
            return True
        except Exception as e:
            print(f"❌ Failed to create experiment: {str(e)}")
            return False
    
    def train_logistic_regression(self):
        """Train Logistic Regression baseline model"""
        print("\n" + "=" * 80)
        print("TRAINING: LOGISTIC REGRESSION")
        print("=" * 80)
        
        try:
            with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
                # Train model
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'  # Handle class imbalance
                )
                model.fit(self.X_train_scaled, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                }
                
                # Parameters
                params = {
                    'max_iter': 1000,
                    'class_weight': 'balanced',
                }
                
                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                # Try to log model, but don't fail if it doesn't work
                try:
                    mlflow.sklearn.log_model(model, "model")
                    print("✓ Model artifact logged to MLflow")
                except Exception as log_error:
                    print(f"  ⚠️  Warning: Could not log model artifact: {str(log_error)}")
                
                # Print results
                print(f"\nMetrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                self.models_trained['LogisticRegression'] = {
                    'model': model,
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print("✓ Model metrics logged to MLflow")
                
        except Exception as e:
            print(f"❌ Error training Logistic Regression: {str(e)}")
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "=" * 80)
        print("TRAINING: RANDOM FOREST")
        print("=" * 80)
        
        try:
            with mlflow.start_run(run_name="Random_Forest"):
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                }
                
                # Parameters
                params = {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'class_weight': 'balanced',
                }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                # Try to log model, but don't fail if it doesn't work
                try:
                    mlflow.sklearn.log_model(model, "model")
                    print("✓ Model artifact logged to MLflow")
                except Exception as log_error:
                    print(f"  ⚠️  Warning: Could not log model artifact: {str(log_error)}")
                
                # Log top features
                try:
                    mlflow.log_text(feature_importance.head(10).to_string(), "top_features.txt")
                except Exception as log_error:
                    print(f"  ⚠️  Warning: Could not log features: {str(log_error)}")
                
                # Print results
                print(f"\nMetrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                print(f"\nTop 10 Important Features:")
                print(feature_importance.head(10).to_string(index=False))
                
                self.models_trained['RandomForest'] = {
                    'model': model,
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'feature_importance': feature_importance
                }
                
                print("✓ Model metrics logged to MLflow")
                
        except Exception as e:
            print(f"❌ Error training Random Forest: {str(e)}")
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n" + "=" * 80)
        print("TRAINING: GRADIENT BOOSTING")
        print("=" * 80)
        
        try:
            with mlflow.start_run(run_name="Gradient_Boosting"):
                # Train model
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    subsample=0.8
                )
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                }
                
                # Parameters
                params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'subsample': 0.8,
                }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                # Try to log model, but don't fail if it doesn't work
                try:
                    mlflow.sklearn.log_model(model, "model")
                    print("✓ Model artifact logged to MLflow")
                except Exception as log_error:
                    print(f"  ⚠️  Warning: Could not log model artifact: {str(log_error)}")
                
                # Log top features
                try:
                    mlflow.log_text(feature_importance.head(10).to_string(), "top_features.txt")
                except Exception as log_error:
                    print(f"  ⚠️  Warning: Could not log features: {str(log_error)}")
                
                # Print results
                print(f"\nMetrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                print(f"\nTop 10 Important Features:")
                print(feature_importance.head(10).to_string(index=False))
                
                self.models_trained['GradientBoosting'] = {
                    'model': model,
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'feature_importance': feature_importance
                }
                
                print("✓ Model metrics logged to MLflow")
                
        except Exception as e:
            print(f"❌ Error training Gradient Boosting: {str(e)}")
    
    def print_model_comparison(self):
        """Print comparison of all trained models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        if not self.models_trained:
            print("❌ No models were trained successfully")
            return None
        
        comparison_data = []
        for model_name, model_info in self.models_trained.items():
            metrics = model_info['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Find best model by ROC-AUC
        if self.models_trained:
            best_model = max(self.models_trained.items(), 
                            key=lambda x: x[1]['metrics']['roc_auc'])
            print(f"\n🏆 Best Model (by ROC-AUC): {best_model[0]}")
            print(f"   ROC-AUC Score: {best_model[1]['metrics']['roc_auc']:.4f}")
        
        return comparison_df
    
    def run(self, features_path='data/processed/features_engineered.csv'):
        """Execute full model training pipeline"""
        print("=" * 80)
        print("STARTING ML MODEL TRAINING PIPELINE")
        print("=" * 80)
        
        try:
            # Load features
            if not self.load_features(features_path):
                return False
            
            # Prepare data
            if not self.prepare_data():
                return False
            
            # Create MLflow experiment
            if not self.create_experiment():
                return False
            
            # Train models
            self.train_logistic_regression()
            self.train_random_forest()
            self.train_gradient_boosting()
            
            # Compare models
            self.print_model_comparison()
            
            print("\n" + "=" * 80)
            print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"\n📊 View results at: {self.mlflow_uri}")
            print(f"📊 Trained models: {len(self.models_trained)}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution function"""
    trainer = ModelTrainer()
    success = trainer.run()
    
    if not success:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

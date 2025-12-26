"""
Feature Engineering Module
Transforms raw patient data into ML-ready features for readmission prediction
"""

import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Database configuration
DB_CONFIG = {
    'host': 'localhost',  # Use 'postgres' when running in Docker, 'localhost' when running locally
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

class FeatureEngineer:
    """Transforms raw healthcare data into ML-ready features"""
    
    def __init__(self, db_config=DB_CONFIG):
        self.db_config = db_config
        self.conn = None
        self.patients_df = None
        self.admissions_df = None
        self.diagnoses_df = None
        self.medications_df = None
        self.features_df = None
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            print(f"Connecting to PostgreSQL at {self.db_config['host']}:{self.db_config['port']}...")
            self.conn = psycopg2.connect(**self.db_config)
            print("✓ Connected to database")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {str(e)}")
            return False
    
    def load_raw_data(self):
        """Load raw data from database"""
        print("\nLoading raw data from database...")
        
        try:
            self.patients_df = pd.read_sql_query("SELECT * FROM patients", self.conn)
            self.admissions_df = pd.read_sql_query("SELECT * FROM admissions", self.conn)
            self.diagnoses_df = pd.read_sql_query("SELECT * FROM diagnoses", self.conn)
            self.medications_df = pd.read_sql_query("SELECT * FROM medications", self.conn)
            
            print(f"✓ Loaded patients: {len(self.patients_df)} rows")
            print(f"✓ Loaded admissions: {len(self.admissions_df)} rows")
            print(f"✓ Loaded diagnoses: {len(self.diagnoses_df)} rows")
            print(f"✓ Loaded medications: {len(self.medications_df)} rows")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load data: {str(e)}")
            return False
    
    def engineer_patient_features(self):
        """Create patient demographic features"""
        print("\nEngineering patient demographic features...")
        
        # Start with patient data
        features = self.patients_df.copy()
        
        # Age groups (binning continuous age into categorical)
        features['age_group'] = pd.cut(
            features['age'],
            bins=[0, 30, 45, 65, 100],
            labels=['18-30', '30-45', '45-65', '65+']
        )
        
        # One-hot encode gender
        features['gender_M'] = (features['gender'] == 'M').astype(int)
        features['gender_F'] = (features['gender'] == 'F').astype(int)
        
        # One-hot encode race
        race_dummies = pd.get_dummies(features['race'], prefix='race', drop_first=False)
        features = pd.concat([features, race_dummies], axis=1)
        
        print(f"✓ Created patient features: {len(features)} rows")
        return features
    
    def engineer_admission_features(self):
        """Create admission-level features"""
        print("Engineering admission-level features...")
        
        admission_features = self.admissions_df.copy()
        
        # Convert dates to datetime if not already
        admission_features['admit_date'] = pd.to_datetime(admission_features['admit_date'])
        admission_features['discharge_date'] = pd.to_datetime(admission_features['discharge_date'])
        
        # Time-based features
        admission_features['admit_month'] = admission_features['admit_date'].dt.month
        admission_features['admit_day_of_week'] = admission_features['admit_date'].dt.dayofweek
        admission_features['admit_quarter'] = admission_features['admit_date'].dt.quarter
        
        # Length of stay is already available, but create derived features
        admission_features['los_category'] = pd.cut(
            admission_features['length_of_stay'],
            bins=[0, 3, 7, 14, 100],
            labels=['short_0-3d', 'medium_3-7d', 'long_7-14d', 'very_long_14+d']
        )
        
        # Binary flags for high-risk length of stay
        admission_features['los_high_risk'] = (admission_features['length_of_stay'] > 7).astype(int)
        
        print(f"✓ Created {len(admission_features)} admission features")
        return admission_features
    
    def engineer_comorbidity_features(self):
        """Create comorbidity and diagnosis complexity features"""
        print("Engineering comorbidity features...")
        
        # Count diagnoses per admission (comorbidity burden)
        comorbidity_counts = self.diagnoses_df.groupby('admission_id').size().reset_index(name='num_diagnoses')
        
        # Create comorbidity severity categories
        comorbidity_counts['comorbidity_burden'] = pd.cut(
            comorbidity_counts['num_diagnoses'],
            bins=[0, 1, 3, 5, 100],
            labels=['low_1', 'moderate_2-3', 'high_4-5', 'very_high_6+']
        )
        
        # Flag for high comorbidity (>= 3 diagnoses)
        comorbidity_counts['high_comorbidity'] = (comorbidity_counts['num_diagnoses'] >= 3).astype(int)
        
        # Identify high-risk diagnoses
        high_risk_codes = [
            'I50.9',      # Heart failure
            'I48.91',     # Atrial fibrillation
            'N18.9',      # Chronic kidney disease
            'J44.9',      # COPD
        ]
        
        high_risk_diags = self.diagnoses_df[
            self.diagnoses_df['icd_code'].isin(high_risk_codes)
        ].drop_duplicates(subset=['admission_id'])
        
        high_risk_flag = high_risk_diags[['admission_id']].copy()
        high_risk_flag['has_high_risk_diagnosis'] = 1
        
        # Merge features
        comorbidity_features = comorbidity_counts.merge(
            high_risk_flag,
            on='admission_id',
            how='left'
        )
        comorbidity_features['has_high_risk_diagnosis'] = comorbidity_features['has_high_risk_diagnosis'].fillna(0).astype(int)
        
        print(f"✓ Created comorbidity features for {len(comorbidity_features)} admissions")
        return comorbidity_features
    
    def engineer_medication_features(self):
        """Create medication complexity and polypharmacy features"""
        print("Engineering medication features...")
        
        # Count medications per admission
        med_counts = self.medications_df.groupby('admission_id').size().reset_index(name='num_medications')
        
        # Polypharmacy indicator (>= 5 medications)
        med_counts['polypharmacy'] = (med_counts['num_medications'] >= 5).astype(int)
        
        # Medication burden categories
        med_counts['medication_burden'] = pd.cut(
            med_counts['num_medications'],
            bins=[-1, 0, 2, 5, 8, 100],
            labels=['none_0', 'low_1-2', 'moderate_3-5', 'high_6-8', 'very_high_9+']
        )
        
        # Specific medication flags (high-risk combinations)
        warfarin_patients = self.medications_df[
            self.medications_df['drug_name'] == 'Warfarin'
        ]['admission_id'].unique()
        
        med_counts['on_warfarin'] = med_counts['admission_id'].isin(warfarin_patients).astype(int)
        
        print(f"✓ Created medication features for {len(med_counts)} admissions")
        return med_counts
    
    def engineer_admission_history_features(self):
        """Create features based on patient admission history"""
        print("Engineering admission history features...")
        
        # Sort admissions by patient and date
        history_df = self.admissions_df.copy()
        history_df['admit_date'] = pd.to_datetime(history_df['admit_date'])
        history_df = history_df.sort_values(['patient_id', 'admit_date'])
        
        # Count previous admissions for each patient (at time of this admission)
        history_df['previous_admissions'] = history_df.groupby('patient_id').cumcount()
        
        # Days since last admission
        history_df['days_since_last_admission'] = history_df.groupby('patient_id')['admit_date'].diff().dt.days
        
        # Admission frequency in past 12 months
        history_df['admit_date_12m_ago'] = history_df['admit_date'] - timedelta(days=365)
        
        # For each admission, count how many previous admissions in past 12 months
        admissions_12m = []
        for idx, row in history_df.iterrows():
            patient_prior = history_df[
                (history_df['patient_id'] == row['patient_id']) &
                (history_df['admit_date'] >= row['admit_date_12m_ago']) &
                (history_df['admit_date'] < row['admit_date'])
            ]
            admissions_12m.append(len(patient_prior))
        
        history_df['admissions_past_12m'] = admissions_12m
        
        # Readmission within 30 days (in past admissions)
        history_df['prior_readmission_30d'] = 0
        for patient in history_df['patient_id'].unique():
            patient_data = history_df[history_df['patient_id'] == patient].sort_values('admit_date')
            for idx in range(1, len(patient_data)):
                current_idx = patient_data.index[idx]
                prev_idx = patient_data.index[idx - 1]
                days_diff = (patient_data.loc[current_idx, 'admit_date'] - 
                           patient_data.loc[prev_idx, 'discharge_date']).days
                if days_diff <= 30:
                    history_df.loc[current_idx, 'prior_readmission_30d'] = 1
        
        print(f"✓ Created admission history features for {len(history_df)} admissions")
        return history_df[['admission_id', 'previous_admissions', 'days_since_last_admission', 
                          'admissions_past_12m', 'prior_readmission_30d']]
    
    def combine_all_features(self):
        """Combine all engineered features into final feature matrix"""
        print("\nCombining all features into final dataset...")
        
        # Start with patient features
        features = self.engineer_patient_features()
        
        # Merge with admission features
        admission_features = self.engineer_admission_features()
        features = features.merge(admission_features, on='patient_id', how='inner')
        
        # Merge with comorbidity features
        comorbidity_features = self.engineer_comorbidity_features()
        features = features.merge(comorbidity_features, on='admission_id', how='left')
        
        # Merge with medication features
        medication_features = self.engineer_medication_features()
        features = features.merge(medication_features, on='admission_id', how='left')
        
        # Merge with admission history features
        history_features = self.engineer_admission_history_features()
        features = features.merge(history_features, on='admission_id', how='left')
        
        # Fill NaN values for missing features (e.g., no medications = 0)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        self.features_df = features
        
        print(f"\n✓ Combined features: {features.shape[0]} rows × {features.shape[1]} columns")
        print(f"✓ Target variable (readmitted_30_days): {features['readmitted_30_days'].sum()} positive cases")
        print(f"✓ Readmission rate: {features['readmitted_30_days'].mean():.2%}")
        
        return features
    
    def save_features(self, output_path='data/processed'):
        """Save engineered features to CSV"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full feature set
        features_file = output_path / 'features_engineered.csv'
        self.features_df.to_csv(features_file, index=False)
        print(f"\n✓ Saved engineered features to {features_file}")
        
        # Save feature info
        feature_cols = [col for col in self.features_df.columns if col not in ['patient_id', 'admission_id', 'created_at']]
        info_file = output_path / 'feature_info.txt'
        with open(info_file, 'w') as f:
            f.write("ENGINEERED FEATURES FOR READMISSION PREDICTION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Features: {len(feature_cols)}\n")
            f.write(f"Total Samples: {len(self.features_df)}\n")
            f.write(f"Target Variable: readmitted_30_days\n")
            f.write(f"Class Balance: {self.features_df['readmitted_30_days'].mean():.2%} positive\n\n")
            f.write("FEATURE CATEGORIES:\n")
            f.write("-" * 60 + "\n")
            f.write("1. Demographics: age, age_group, gender_*, race_*\n")
            f.write("2. Admission Timing: admit_month, admit_day_of_week, admit_quarter\n")
            f.write("3. Length of Stay: length_of_stay, los_category, los_high_risk\n")
            f.write("4. Comorbidities: num_diagnoses, comorbidity_burden, high_comorbidity, has_high_risk_diagnosis\n")
            f.write("5. Medications: num_medications, polypharmacy, medication_burden, on_warfarin\n")
            f.write("6. Admission History: previous_admissions, days_since_last_admission, admissions_past_12m, prior_readmission_30d\n")
        
        print(f"✓ Saved feature information to {info_file}")
        
        return features_file
    
    def get_feature_summary(self):
        """Print summary of engineered features"""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 80)
        print(f"\nTotal Samples: {len(self.features_df):,}")
        print(f"Total Features: {len(self.features_df.columns)}")
        print(f"\nTarget Variable Distribution:")
        print(f"  Readmitted (1): {self.features_df['readmitted_30_days'].sum():,} ({self.features_df['readmitted_30_days'].mean():.2%})")
        print(f"  Not Readmitted (0): {(self.features_df['readmitted_30_days'] == 0).sum():,} ({(1 - self.features_df['readmitted_30_days'].mean()):.2%})")
        
        print(f"\nFeature Data Types:")
        print(f"  Numeric: {len(self.features_df.select_dtypes(include=[np.number]).columns)}")
        print(f"  Categorical: {len(self.features_df.select_dtypes(include=['object']).columns)}")
        print(f"  Boolean: {len(self.features_df.select_dtypes(include=['bool']).columns)}")
        
        print(f"\nMissing Values: {self.features_df.isnull().sum().sum()}")
        print("\n" + "=" * 80)
    
    def run(self, output_path='data/processed'):
        """Execute full feature engineering pipeline"""
        print("=" * 80)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        try:
            if not self.connect():
                return False
            
            if not self.load_raw_data():
                return False
            
            self.combine_all_features()
            self.get_feature_summary()
            self.save_features(output_path)
            
            print("\n✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if self.conn:
                self.conn.close()
                print("✓ Database connection closed")


def main():
    """Main execution function"""
    engineer = FeatureEngineer()
    success = engineer.run()
    
    if not success:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

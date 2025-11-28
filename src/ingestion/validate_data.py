"""
Data Validation Module
Validates quality of generated patient data before loading to database
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Configuration
DATA_DIR = Path("data/raw")

class DataValidator:
    """Validates healthcare data quality"""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.errors = []
        self.warnings = []
        
    def load_data(self):
        """Load all CSV files"""
        print("Loading data files...")
        try:
            self.patients = pd.read_csv(self.data_dir / "patients.csv")
            self.admissions = pd.read_csv(self.data_dir / "admissions.csv")
            self.diagnoses = pd.read_csv(self.data_dir / "diagnoses.csv")
            self.medications = pd.read_csv(self.data_dir / "medications.csv")
            print(f"✓ Loaded: {len(self.patients)} patients, {len(self.admissions)} admissions")
            return True
        except Exception as e:
            self.errors.append(f"Failed to load data: {str(e)}")
            return False
    
    def validate_patients(self):
        """Validate patient data"""
        print("\nValidating patients table...")
        
        # Check required fields
        required_fields = ['patient_id', 'age', 'gender', 'race']
        for field in required_fields:
            if field not in self.patients.columns:
                self.errors.append(f"Missing required field in patients: {field}")
                return
            
            null_count = self.patients[field].isnull().sum()
            if null_count > 0:
                self.errors.append(f"Found {null_count} null values in patients.{field}")
        
        # Check age range
        if self.patients['age'].min() < 18:
            self.errors.append(f"Found patients under 18: min age = {self.patients['age'].min()}")
        if self.patients['age'].max() > 100:
            self.warnings.append(f"Found patients over 100: max age = {self.patients['age'].max()}")
        
        # Check for duplicates
        duplicates = self.patients['patient_id'].duplicated().sum()
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate patient_ids")
        
        # Check valid genders
        valid_genders = ['M', 'F']
        invalid_genders = ~self.patients['gender'].isin(valid_genders)
        if invalid_genders.any():
            self.errors.append(f"Found {invalid_genders.sum()} invalid gender values")
        
        print(f"✓ Patients validation: {len(self.patients)} records")
    
    def validate_admissions(self):
        """Validate admissions data"""
        print("\nValidating admissions table...")
        
        # Convert date columns to datetime
        self.admissions['admit_date'] = pd.to_datetime(self.admissions['admit_date'])
        self.admissions['discharge_date'] = pd.to_datetime(self.admissions['discharge_date'])
        
        # Check required fields
        required_fields = ['admission_id', 'patient_id', 'admit_date', 'discharge_date', 'length_of_stay']
        for field in required_fields:
            if field not in self.admissions.columns:
                self.errors.append(f"Missing required field in admissions: {field}")
                return
            
            null_count = self.admissions[field].isnull().sum()
            if null_count > 0:
                self.errors.append(f"Found {null_count} null values in admissions.{field}")
        
        # Check date logic: discharge must be after admit
        invalid_dates = self.admissions['discharge_date'] <= self.admissions['admit_date']
        if invalid_dates.any():
            self.errors.append(f"Found {invalid_dates.sum()} admissions with discharge_date <= admit_date")
        
        # Check length of stay consistency
        calculated_los = (self.admissions['discharge_date'] - self.admissions['admit_date']).dt.days
        los_mismatch = (calculated_los != self.admissions['length_of_stay']).sum()
        if los_mismatch > 0:
            self.warnings.append(f"Found {los_mismatch} admissions with LOS mismatch")
        
        # Check LOS range
        if self.admissions['length_of_stay'].min() < 1:
            self.errors.append(f"Found admissions with LOS < 1 day")
        if self.admissions['length_of_stay'].max() > 365:
            self.warnings.append(f"Found admissions with LOS > 365 days: max = {self.admissions['length_of_stay'].max()}")
        
        # Check referential integrity: all patient_ids must exist in patients
        invalid_patients = ~self.admissions['patient_id'].isin(self.patients['patient_id'])
        if invalid_patients.any():
            self.errors.append(f"Found {invalid_patients.sum()} admissions with invalid patient_id")
        
        # Check for duplicate admission_ids
        duplicates = self.admissions['admission_id'].duplicated().sum()
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate admission_ids")
        
        print(f"✓ Admissions validation: {len(self.admissions)} records")
    
    def validate_diagnoses(self):
        """Validate diagnoses data"""
        print("\nValidating diagnoses table...")
        
        # Check required fields
        required_fields = ['diagnosis_id', 'admission_id', 'icd_code', 'description']
        for field in required_fields:
            if field not in self.diagnoses.columns:
                self.errors.append(f"Missing required field in diagnoses: {field}")
                return
            
            null_count = self.diagnoses[field].isnull().sum()
            if null_count > 0:
                self.errors.append(f"Found {null_count} null values in diagnoses.{field}")
        
        # Check referential integrity: all admission_ids must exist
        invalid_admissions = ~self.diagnoses['admission_id'].isin(self.admissions['admission_id'])
        if invalid_admissions.any():
            self.errors.append(f"Found {invalid_admissions.sum()} diagnoses with invalid admission_id")
        
        # Check for duplicate diagnosis_ids
        duplicates = self.diagnoses['diagnosis_id'].duplicated().sum()
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate diagnosis_ids")
        
        print(f"✓ Diagnoses validation: {len(self.diagnoses)} records")
    
    def validate_medications(self):
        """Validate medications data"""
        print("\nValidating medications table...")
        
        # Check required fields
        required_fields = ['medication_id', 'admission_id', 'drug_name', 'dosage']
        for field in required_fields:
            if field not in self.medications.columns:
                self.errors.append(f"Missing required field in medications: {field}")
                return
            
            null_count = self.medications[field].isnull().sum()
            if null_count > 0:
                self.errors.append(f"Found {null_count} null values in medications.{field}")
        
        # Check referential integrity: all admission_ids must exist
        invalid_admissions = ~self.medications['admission_id'].isin(self.admissions['admission_id'])
        if invalid_admissions.any():
            self.errors.append(f"Found {invalid_admissions.sum()} medications with invalid admission_id")
        
        # Check for duplicate medication_ids
        duplicates = self.medications['medication_id'].duplicated().sum()
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate medication_ids")
        
        print(f"✓ Medications validation: {len(self.medications)} records")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ ALL CHECKS PASSED - Data quality is excellent!")
        elif not self.errors:
            print("\n✅ VALIDATION PASSED - Data has warnings but no critical errors")
        else:
            print("\n❌ VALIDATION FAILED - Critical errors found")
        
        print("=" * 60)
        
        return len(self.errors) == 0
    
    def validate_all(self):
        """Run all validation checks"""
        print("=" * 60)
        print("STARTING DATA VALIDATION")
        print("=" * 60)
        
        if not self.load_data():
            self.print_summary()
            return False
        
        self.validate_patients()
        self.validate_admissions()
        self.validate_diagnoses()
        self.validate_medications()
        
        return self.print_summary()


def main():
    """Main validation function"""
    validator = DataValidator()
    success = validator.validate_all()
    
    if not success:
        sys.exit(1)  # Exit with error code if validation fails
    
    sys.exit(0)  # Exit successfully


if __name__ == "__main__":
    main()
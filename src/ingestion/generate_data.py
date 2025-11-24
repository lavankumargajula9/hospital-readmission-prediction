"""
Synthetic Patient Data Generator
Generates realistic healthcare data for portfolio project
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_PATIENTS = 1000
AVG_ADMISSIONS_PER_PATIENT = 2
OUTPUT_DIR = Path("data/raw")

# Reference data (realistic medical codes and names)
ICD_CODES = [
    ("I50.9", "Heart failure, unspecified"),
    ("I10", "Essential hypertension"),
    ("E11.9", "Type 2 diabetes without complications"),
    ("J44.9", "COPD, unspecified"),
    ("N18.9", "Chronic kidney disease, unspecified"),
    ("I48.91", "Atrial fibrillation"),
    ("F17.210", "Nicotine dependence"),
    ("E78.5", "Hyperlipidemia"),
    ("J18.9", "Pneumonia, unspecified"),
    ("I25.10", "Coronary artery disease")
]

MEDICATIONS = [
    "Lisinopril", "Metformin", "Amlodipine", "Metoprolol", 
    "Omeprazole", "Simvastatin", "Losartan", "Furosemide",
    "Warfarin", "Insulin"
]

GENDERS = ["M", "F"]
RACES = ["White", "Black", "Hispanic", "Asian", "Other"]


def generate_patients(num_patients):
    """Generate patient demographics"""
    
    print(f"Generating {num_patients} patients...")
    
    patients = pd.DataFrame({
        'patient_id': range(1, num_patients + 1),
        'age': np.random.normal(loc=65, scale=15, size=num_patients).clip(18, 100).astype(int),
        'gender': np.random.choice(GENDERS, size=num_patients),
        'race': np.random.choice(RACES, size=num_patients, p=[0.6, 0.15, 0.15, 0.05, 0.05]),
        'created_at': datetime.now()
    })
    
    return patients


def generate_admissions(patients_df, avg_admissions):
    """Generate hospital admissions for patients"""
    
    # Number of admissions per patient (Poisson distribution)
    num_admissions = np.random.poisson(lam=avg_admissions, size=len(patients_df))
    
    admissions_list = []
    admission_id = 1
    
    print(f"Generating ~{len(patients_df) * avg_admissions} admissions...")
    
    for idx, patient_id in enumerate(patients_df['patient_id']):
        for _ in range(num_admissions[idx]):
            # Random admission in past 2 years
            admit_date = datetime.now() - timedelta(days=random.randint(0, 730))
            
            # Length of stay: 1-30 days
            los = random.randint(1, 30)
            discharge_date = admit_date + timedelta(days=los)
            
            # 30-day readmission flag (15% chance)
            is_readmission = random.random() < 0.15
            
            admissions_list.append({
                'admission_id': admission_id,
                'patient_id': patient_id,
                'admit_date': admit_date,
                'discharge_date': discharge_date,
                'length_of_stay': los,
                'readmitted_30_days': is_readmission,
                'created_at': datetime.now()
            })
            
            admission_id += 1
    
    return pd.DataFrame(admissions_list)


def generate_diagnoses(admissions_df):
    """Generate diagnoses for each admission"""
    
    diagnoses_list = []
    diagnosis_id = 1
    
    print(f"Generating diagnoses...")
    
    for admission_id in admissions_df['admission_id']:
        # Each admission has 1-5 diagnoses
        num_diagnoses = random.randint(1, 5)
        
        # Sample diagnoses without replacement
        selected_diagnoses = random.sample(ICD_CODES, min(num_diagnoses, len(ICD_CODES)))
        
        for icd_code, description in selected_diagnoses:
            diagnoses_list.append({
                'diagnosis_id': diagnosis_id,
                'admission_id': admission_id,
                'icd_code': icd_code,
                'description': description,
                'created_at': datetime.now()
            })
            diagnosis_id += 1
    
    return pd.DataFrame(diagnoses_list)


def generate_medications(admissions_df):
    """Generate medications for each admission"""
    
    medications_list = []
    medication_id = 1
    
    print(f"Generating medications...")
    
    for admission_id in admissions_df['admission_id']:
        # Each admission has 0-8 medications
        num_medications = random.randint(0, 8)
        
        selected_meds = random.sample(MEDICATIONS, min(num_medications, len(MEDICATIONS)))
        
        for med_name in selected_meds:
            dosage = f"{random.choice([5, 10, 20, 40, 50, 100])}mg"
            
            medications_list.append({
                'medication_id': medication_id,
                'admission_id': admission_id,
                'drug_name': med_name,
                'dosage': dosage,
                'created_at': datetime.now()
            })
            medication_id += 1
    
    return pd.DataFrame(medications_list)


def save_data(patients, admissions, diagnoses, medications, output_dir):
    """Save all dataframes to CSV files"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving data to {output_dir}...")
    
    patients.to_csv(output_dir / "patients.csv", index=False)
    print(f"✓ Saved {len(patients)} patients")
    
    admissions.to_csv(output_dir / "admissions.csv", index=False)
    print(f"✓ Saved {len(admissions)} admissions")
    
    diagnoses.to_csv(output_dir / "diagnoses.csv", index=False)
    print(f"✓ Saved {len(diagnoses)} diagnoses")
    
    medications.to_csv(output_dir / "medications.csv", index=False)
    print(f"✓ Saved {len(medications)} medications")


def main():
    """Main execution function"""
    
    print("=" * 50)
    print("SYNTHETIC PATIENT DATA GENERATOR")
    print("=" * 50)
    
    # Generate data
    patients = generate_patients(NUM_PATIENTS)
    admissions = generate_admissions(patients, AVG_ADMISSIONS_PER_PATIENT)
    diagnoses = generate_diagnoses(admissions)
    medications = generate_medications(admissions)
    
    # Save to CSV
    save_data(patients, admissions, diagnoses, medications, OUTPUT_DIR)
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE!")
    print("=" * 50)
    print(f"Total Patients: {len(patients)}")
    print(f"Total Admissions: {len(admissions)}")
    print(f"Readmission Rate: {admissions['readmitted_30_days'].mean():.1%}")
    print(f"Avg Length of Stay: {admissions['length_of_stay'].mean():.1f} days")
    print(f"Total Diagnoses: {len(diagnoses)}")
    print(f"Total Medications: {len(medications)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
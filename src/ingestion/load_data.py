"""
Data Loading Module
Loads validated patient data from CSV into PostgreSQL database
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
import sys

# Configuration
DATA_DIR = Path("data/raw")
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

class DataLoader:
    """Loads healthcare data into PostgreSQL"""
    
    def __init__(self, db_config=DB_CONFIG, data_dir=DATA_DIR):
        self.db_config = db_config
        self.data_dir = data_dir
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            print(f"Connecting to PostgreSQL at {self.db_config['host']}:{self.db_config['port']}...")
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("✓ Connected to database")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {str(e)}")
            return False
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        print("\nCreating tables...")
        
        # Patients table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY,
                age INTEGER NOT NULL,
                gender VARCHAR(1) NOT NULL,
                race VARCHAR(50) NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        print("✓ Created/verified patients table")
        
        # Admissions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS admissions (
                admission_id INTEGER PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                admit_date TIMESTAMP NOT NULL,
                discharge_date TIMESTAMP NOT NULL,
                length_of_stay INTEGER NOT NULL,
                readmitted_30_days BOOLEAN NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        print("✓ Created/verified admissions table")
        
        # Diagnoses table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnoses (
                diagnosis_id INTEGER PRIMARY KEY,
                admission_id INTEGER NOT NULL,
                icd_code VARCHAR(20) NOT NULL,
                description TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
            )
        """)
        print("✓ Created/verified diagnoses table")
        
        # Medications table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                medication_id INTEGER PRIMARY KEY,
                admission_id INTEGER NOT NULL,
                drug_name VARCHAR(100) NOT NULL,
                dosage VARCHAR(50) NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
            )
        """)
        print("✓ Created/verified medications table")
        
        self.conn.commit()
    
    def truncate_tables(self):
        """Truncate all tables for fresh load"""
        print("\nTruncating existing data...")
        
        # Truncate in reverse order due to foreign keys
        tables = ['medications', 'diagnoses', 'admissions', 'patients']
        for table in tables:
            self.cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
            print(f"✓ Truncated {table}")
        
        self.conn.commit()
    
    def load_patients(self):
        """Load patients data"""
        print("\nLoading patients...")
        
        df = pd.read_csv(self.data_dir / "patients.csv")
        
        # Convert to list of tuples
        data = [tuple(row) for row in df.values]
        
        # Bulk insert
        execute_values(
            self.cursor,
            "INSERT INTO patients (patient_id, age, gender, race, created_at) VALUES %s",
            data
        )
        
        self.conn.commit()
        
        # Verify count
        self.cursor.execute("SELECT COUNT(*) FROM patients")
        count = self.cursor.fetchone()[0]
        print(f"✓ Loaded {count} patients")
        
        return count
    
    def load_admissions(self):
        """Load admissions data"""
        print("\nLoading admissions...")
        
        df = pd.read_csv(self.data_dir / "admissions.csv")
        
        # Convert to list of tuples
        data = [tuple(row) for row in df.values]
        
        # Bulk insert
        execute_values(
            self.cursor,
            """INSERT INTO admissions 
               (admission_id, patient_id, admit_date, discharge_date, 
                length_of_stay, readmitted_30_days, created_at) 
               VALUES %s""",
            data
        )
        
        self.conn.commit()
        
        # Verify count
        self.cursor.execute("SELECT COUNT(*) FROM admissions")
        count = self.cursor.fetchone()[0]
        print(f"✓ Loaded {count} admissions")
        
        return count
    
    def load_diagnoses(self):
        """Load diagnoses data"""
        print("\nLoading diagnoses...")
        
        df = pd.read_csv(self.data_dir / "diagnoses.csv")
        
        # Convert to list of tuples
        data = [tuple(row) for row in df.values]
        
        # Bulk insert
        execute_values(
            self.cursor,
            """INSERT INTO diagnoses 
               (diagnosis_id, admission_id, icd_code, description, created_at) 
               VALUES %s""",
            data
        )
        
        self.conn.commit()
        
        # Verify count
        self.cursor.execute("SELECT COUNT(*) FROM diagnoses")
        count = self.cursor.fetchone()[0]
        print(f"✓ Loaded {count} diagnoses")
        
        return count
    
    def load_medications(self):
        """Load medications data"""
        print("\nLoading medications...")
        
        df = pd.read_csv(self.data_dir / "medications.csv")
        
        # Convert to list of tuples
        data = [tuple(row) for row in df.values]
        
        # Bulk insert
        execute_values(
            self.cursor,
            """INSERT INTO medications 
               (medication_id, admission_id, drug_name, dosage, created_at) 
               VALUES %s""",
            data
        )
        
        self.conn.commit()
        
        # Verify count
        self.cursor.execute("SELECT COUNT(*) FROM medications")
        count = self.cursor.fetchone()[0]
        print(f"✓ Loaded {count} medications")
        
        return count
    
    def print_summary(self, counts):
        """Print loading summary"""
        print("\n" + "=" * 60)
        print("LOADING SUMMARY")
        print("=" * 60)
        print(f"Patients:    {counts['patients']:,}")
        print(f"Admissions:  {counts['admissions']:,}")
        print(f"Diagnoses:   {counts['diagnoses']:,}")
        print(f"Medications: {counts['medications']:,}")
        print("=" * 60)
        print("✅ DATA SUCCESSFULLY LOADED TO POSTGRESQL")
        print("=" * 60)
    
    def load_all(self):
        """Load all data"""
        print("=" * 60)
        print("STARTING DATA LOAD TO POSTGRESQL")
        print("=" * 60)
        
        try:
            if not self.connect():
                return False
            
            self.create_tables()
            self.truncate_tables()
            
            counts = {
                'patients': self.load_patients(),
                'admissions': self.load_admissions(),
                'diagnoses': self.load_diagnoses(),
                'medications': self.load_medications()
            }
            
            self.print_summary(counts)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
            
        finally:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
                print("\n✓ Database connection closed")


def main():
    """Main loading function"""
    loader = DataLoader()
    success = loader.load_all()
    
    if not success:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
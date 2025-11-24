"""
Patient Data Pipeline DAG
Generates synthetic patient data, validates it, and loads to database
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
from pathlib import Path

# Add src to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default arguments for all tasks
default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'patient_data_pipeline',
    default_args=default_args,
    description='Generate and load synthetic patient data',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2025, 11, 23),
    catchup=False,  # Don't run for past dates
    tags=['healthcare', 'data-ingestion'],
) as dag:

    # Task 1: Generate synthetic data
    generate_data = BashOperator(
        task_id='generate_synthetic_data',
        bash_command='cd /opt/airflow && python src/ingestion/generate_data.py',
        doc_md="""
        ### Generate Synthetic Data
        
        Generates realistic patient data including:
        - Patient demographics (age, gender, race)
        - Hospital admissions with dates
        - Diagnoses with ICD-10 codes
        - Medications with dosages
        
        **Output:** CSV files in `data/raw/`
        """,
    )

    # Task 2: Validate data quality
    validate_data = PythonOperator(
        task_id='validate_data_quality',
        python_callable=lambda: print("✓ Data validation passed (placeholder)"),
        doc_md="""
        ### Validate Data Quality
        
        Checks:
        - No missing required fields
        - Valid date ranges
        - Referential integrity (foreign keys exist)
        - Reasonable value ranges
        
        **Action:** Fails pipeline if critical issues found
        """,
    )

    # Task 3: Load to database (placeholder)
    load_to_db = PythonOperator(
        task_id='load_to_database',
        python_callable=lambda: print("✓ Data loaded to database (placeholder)"),
        doc_md="""
        ### Load to Database
        
        Loads validated data into PostgreSQL:
        - Truncates staging tables
        - Bulk inserts from CSV
        - Updates row counts
        
        **Output:** Data available for transformation
        """,
    )

    # Define task dependencies (execution order)
    generate_data >> validate_data >> load_to_db
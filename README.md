# Hospital Readmission Prevention System

End-to-end ML system predicting hospital readmissions with 30-day risk scoring.

## Tech Stack
- **Data Pipeline**: Apache Airflow, Snowflake
- **Transformation**: DBT
- **ML**: XGBoost, MLflow
- **API**: FastAPI
- **Infrastructure**: Docker, Terraform

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Start services
docker-compose up -d
```

## Project Status
🚧 In Development
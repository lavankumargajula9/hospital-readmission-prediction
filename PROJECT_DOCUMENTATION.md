# Hospital Readmission Prevention System - Project Documentation

**Project Date:** November 2025 - December 2025
**Status:** Baseline Model Training Complete
**Last Updated:** December 25, 2025

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Results & Performance](#results--performance)
8. [Findings & Insights](#findings--insights)
9. [Future Improvements](#future-improvements)
10. [Technical Stack](#technical-stack)

---

## Project Overview

### Objective
Build an end-to-end machine learning system to predict hospital readmission within 30 days of discharge. The system demonstrates production-grade data engineering, feature engineering, and ML lifecycle management using industry-standard tools.

### Key Metrics
- **Dataset Size:** 1,000 patients with 1,994 hospital admissions
- **Readmission Rate:** 15.2% (302 readmitted / 1,692 not readmitted)
- **Feature Count:** 29 engineered features
- **Models Trained:** 3 (Logistic Regression, Random Forest, Gradient Boosting)
- **Baseline ROC-AUC:** 0.5781 (Logistic Regression)

### Use Case
Healthcare systems need to identify high-risk patients for intervention programs. Early readmission prevention can reduce costs, improve outcomes, and enhance patient care quality.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    HOSPITAL READMISSION SYSTEM               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐        ┌──────────────┐                   │
│  │   Airflow    │        │  PostgreSQL  │                   │
│  │   DAG        │◄──────►│  Database    │                   │
│  │  Orchestrator│        │              │                   │
│  └──────────────┘        └──────────────┘                   │
│         │                       │                            │
│         │ Schedules             │ Stores                     │
│         ▼                       ▼                            │
│  ┌──────────────────────────────────┐                       │
│  │   Data Ingestion Pipeline        │                       │
│  │ - Generate synthetic data        │                       │
│  │ - Validate data quality          │                       │
│  │ - Load to PostgreSQL             │                       │
│  └──────────────────────────────────┘                       │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────┐                       │
│  │  Feature Engineering Pipeline    │                       │
│  │ - Demographics features          │                       │
│  │ - Admission features             │                       │
│  │ - Comorbidity features           │                       │
│  │ - Medication features            │                       │
│  │ - History features               │                       │
│  └──────────────────────────────────┘                       │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────┐                       │
│  │   ML Training Pipeline           │                       │
│  │ - Logistic Regression            │                       │
│  │ - Random Forest                  │                       │
│  │ - Gradient Boosting              │                       │
│  │ - MLflow Tracking                │                       │
│  └──────────────────────────────────┘                       │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────┐                       │
│  │    MLflow Model Registry         │                       │
│  │ - Experiment tracking            │                       │
│  │ - Metrics logging                │                       │
│  │ - Model versioning               │                       │
│  └──────────────────────────────────┘                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Docker Services
- **PostgreSQL 13:** Data storage (port 5432)
- **Redis 7.2:** Message broker (port 6379)
- **Apache Airflow 2.7.0:** Workflow orchestration (port 8080)
- **MLflow 2.7.0:** Model tracking (port 5000)

---

## Data Pipeline

### Data Generation

**Script:** `src/ingestion/generate_data.py`

#### Synthetic Data Characteristics
- **1,000 patients** with realistic demographics
- **~2,000 admissions** (average 2 per patient, Poisson distributed)
- **ICD-10 diagnoses** (10 common conditions)
- **10 medications** with realistic dosages
- **30-day readmission rate: 15%** (realistic for healthcare)

#### Generated Tables

**Patients Table**
- patient_id: Unique identifier
- age: 18-100 years, normally distributed (mean=65)
- gender: Male/Female (60% White, 15% Black, 15% Hispanic, 5% Asian, 5% Other)
- race: Ethnicity categories
- created_at: Record creation timestamp

**Admissions Table**
- admission_id: Unique identifier
- patient_id: Foreign key to patients
- admit_date: Hospital admission date (past 2 years)
- discharge_date: Hospital discharge date
- length_of_stay: 1-30 days
- readmitted_30_days: Binary flag (15% positive)
- created_at: Record creation timestamp

**Diagnoses Table**
- diagnosis_id: Unique identifier
- admission_id: Foreign key to admissions
- icd_code: ICD-10 code (e.g., I50.9 - Heart failure)
- description: Diagnosis description
- created_at: Record creation timestamp

**Medications Table**
- medication_id: Unique identifier
- admission_id: Foreign key to admissions
- drug_name: Medication name (e.g., Lisinopril, Metformin)
- dosage: Dosage amount (e.g., 10mg, 20mg)
- created_at: Record creation timestamp

### Data Validation

**Script:** `src/ingestion/validate_data.py`

Validation checks performed:
- ✅ No missing required fields
- ✅ Valid date ranges (discharge > admit)
- ✅ Referential integrity (patient_ids exist)
- ✅ Reasonable value ranges (age 18-100, LOS > 0)
- ✅ No duplicate IDs
- ✅ Proper data types

**Result:** All checks passed - data quality is excellent

### Data Loading

**Script:** `src/ingestion/load_data.py`

- Connects to PostgreSQL
- Creates tables if not exist
- Truncates staging tables
- Bulk inserts using execute_values (optimized for performance)
- Verifies row counts

**Final Data Summary:**
- Patients: 1,000
- Admissions: 1,994
- Diagnoses: ~5,000 records
- Medications: ~4,000 records

---

## Exploratory Data Analysis

**Notebook:** `hospital_eda.ipynb`

### Patient Demographics

**Age Distribution**
- Mean: 65.1 years
- Median: 65 years
- Std Dev: 15.3 years
- Range: 18-100 years
- Distribution: Approximately normal (realistic for healthcare)

**Gender Distribution**
- Male: 49.8% (498)
- Female: 50.2% (502)
- Balanced dataset

**Race/Ethnicity Distribution**
- White: 60.0% (600)
- Black: 15.0% (150)
- Hispanic: 15.0% (150)
- Asian: 5.0% (50)
- Other: 5.0% (50)

### Admission Characteristics

**Total Admissions:** 1,994

**Length of Stay**
- Mean: 10.5 days
- Median: 10 days
- Range: 1-30 days
- High variability (realistic)

**Readmission Analysis**
- 30-day readmission rate: 15.2% (302/1,994)
- Not readmitted: 84.8% (1,692/1,994)
- Slightly imbalanced but realistic

**Length of Stay by Readmission Status**
- Not readmitted: Mean = 10.2 days
- Readmitted: Mean = 11.3 days
- Readmitted patients have slightly longer stays (potential predictor)

### Diagnosis Patterns

**Top 5 Diagnoses**
1. Heart Failure (I50.9): ~583 cases
2. Essential Hypertension (I10): ~572 cases
3. Type 2 Diabetes (E11.9): ~562 cases
4. COPD (J44.9): ~549 cases
5. Chronic Kidney Disease (N18.9): ~543 cases

**Diagnoses per Admission**
- Average: 2.5 diagnoses per admission
- Range: 1-5 diagnoses
- Indicator of disease complexity

### Medication Patterns

**Average Medications per Admission:** 2.0

**Most Prescribed Medications**
- Lisinopril (ACE inhibitor): ~389 cases
- Metformin (diabetes): ~368 cases
- Amlodipine (calcium channel blocker): ~365 cases
- Metoprolol (beta blocker): ~358 cases
- Omeprazole (PPI): ~343 cases

### Key Insights

1. **Age is a relevant factor** - Mean age is 65, typical for high-risk population
2. **Chronic disease burden** - High prevalence of cardiac, metabolic, renal diseases
3. **Medication complexity** - Multiple medications per admission indicate severity
4. **Readmission risk exists** - 15% readmission rate gives clear positive class
5. **Data quality is excellent** - No missing values, proper referential integrity

---

## Feature Engineering

**Script:** `src/features/feature_engineering.py`

### Features Engineered

Total features created: **29**

#### 1. Demographics Features (5)
- `age`: Continuous age in years
- `age_group`: Categorical (18-30, 30-45, 45-65, 65+)
- `gender_M`: Binary (male=1, female=0)
- `gender_F`: Binary (female=1, male=0)
- `race_*`: One-hot encoded race categories (Asian, Black, Hispanic, Other, White)

#### 2. Admission Timing Features (3)
- `admit_month`: Month of admission (1-12)
- `admit_day_of_week`: Day of week (0-6)
- `admit_quarter`: Quarter of year (1-4)

#### 3. Length of Stay Features (3)
- `length_of_stay`: Continuous days in hospital
- `los_category`: Categorical (short_0-3d, medium_3-7d, long_7-14d, very_long_14+d)
- `los_high_risk`: Binary (>7 days = 1)

#### 4. Comorbidity Features (4)
- `num_diagnoses`: Count of diagnoses (continuous)
- `comorbidity_burden`: Categorical (low_1, moderate_2-3, high_4-5, very_high_6+)
- `high_comorbidity`: Binary (>=3 diagnoses = 1)
- `has_high_risk_diagnosis`: Binary flag for cardiac/renal/COPD diagnoses

#### 5. Medication Features (4)
- `num_medications`: Count of medications (continuous)
- `polypharmacy`: Binary (>=5 medications = 1)
- `medication_burden`: Categorical (none_0, low_1-2, moderate_3-5, high_6-8, very_high_9+)
- `on_warfarin`: Binary (anticoagulant use)

#### 6. Admission History Features (4)
- `previous_admissions`: Count of prior admissions for patient
- `days_since_last_admission`: Days between current and previous admission
- `admissions_past_12m`: Number of admissions in past 12 months
- `prior_readmission_30d`: Binary (had prior 30-day readmission)

### Feature Engineering Process

1. **Load raw data** from PostgreSQL
2. **Create demographic features** - Age binning, gender/race encoding
3. **Create temporal features** - Extraction from dates
4. **Create comorbidity features** - Aggregation of diagnoses
5. **Create medication features** - Aggregation and polypharmacy flags
6. **Create history features** - Patient-level temporal patterns
7. **Combine all features** - Merge into single feature matrix
8. **Handle missing values** - Fill with 0 (appropriate for count features)

### Output

**File:** `data/processed/features_engineered.csv`
- Rows: 1,994 (one per admission)
- Columns: 36 (including target and identifiers)
- Target variable: `readmitted_30_days` (0/1)

---

## Model Training

**Script:** `src/modeling/train_model.py`

### Data Preparation

**Train/Test Split:** 80/20 stratified
- Training set: 1,595 admissions (80%)
- Test set: 399 admissions (20%)
- Stratification preserves readmission rate in both sets

**Feature Scaling:** StandardScaler
- Applied to all numeric features
- Scaled separately on train and test sets
- Necessary for Logistic Regression, improves tree-based models

**Categorical Encoding:** One-hot encoding
- Encoded categorical features (gender, race, created_at, admit_date, discharge_date)
- Dropped first category to avoid multicollinearity
- Final feature count: 29 numeric features

### Models Trained

#### 1. Logistic Regression (Baseline)
**Purpose:** Fast, interpretable baseline model

**Hyperparameters:**
- max_iter: 1000
- class_weight: 'balanced' (handle class imbalance)
- solver: default (lbfgs)

**Performance:**
- Accuracy: 0.8471
- Precision: 0.3103
- Recall: 0.0993
- F1: 0.1514
- ROC-AUC: **0.5781**

**Interpretation:**
- High accuracy but misleading (can achieve 84.7% by predicting all negatives)
- Very low recall (catches only 9.9% of readmissions)
- ROC-AUC of 0.5781 indicates weak discriminative power (barely better than random)

#### 2. Random Forest
**Purpose:** Capture non-linear relationships and feature interactions

**Hyperparameters:**
- n_estimators: 100 trees
- max_depth: 15
- min_samples_split: 10
- min_samples_leaf: 5
- class_weight: 'balanced'
- n_jobs: -1 (parallel processing)

**Performance:**
- Accuracy: 0.8471
- Precision: 0.3333
- Recall: 0.0993
- F1: 0.1538
- ROC-AUC: 0.5733

**Interpretation:**
- Similar performance to Logistic Regression
- Slight improvement in precision but slight decrease in ROC-AUC
- Feature importance shows which features matter most

#### 3. Gradient Boosting
**Purpose:** Sequential ensemble learning with strong baseline

**Hyperparameters:**
- n_estimators: 100 boosting rounds
- learning_rate: 0.1
- max_depth: 5
- min_samples_split: 10
- min_samples_leaf: 5
- subsample: 0.8

**Performance:**
- Accuracy: 0.8471
- Precision: 0.3750
- Recall: 0.1987
- F1: 0.2590
- ROC-AUC: 0.5755

**Interpretation:**
- Best recall (catches 19.9% of readmissions)
- Best F1 score (0.2590)
- Still low ROC-AUC indicates feature set is weak

### MLflow Integration

**Tracking:** All experiments logged to MLflow
- Experiment: `Hospital_Readmission_Prediction`
- Runs: 3 (one per model)
- Logged: Parameters, metrics, model artifacts
- UI: http://localhost:5000

---

## Results & Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.8471 | 0.3103 | 0.0993 | 0.1514 | **0.5781** |
| Random Forest | 0.8471 | 0.3333 | 0.0993 | 0.1538 | 0.5733 |
| Gradient Boosting | 0.8471 | 0.3750 | 0.1987 | 0.2590 | 0.5755 |

### Key Findings

**Best Baseline Model:** Logistic Regression (by ROC-AUC)
- ROC-AUC: 0.5781
- Interpretable coefficients
- Fast to train and deploy

**Most Promising for Production:** Gradient Boosting
- Best recall (catches more readmissions)
- Best F1 score (balanced performance)
- Better for intervention targeting

### Performance Assessment

**Current Performance:** ⚠️ Weak
- ROC-AUC of 0.5781 is barely above random (0.5)
- Models struggle to distinguish readmission risk
- Feature set is insufficient for good prediction

**Why Performance is Low:**
1. Features may not capture readmission drivers
2. Synthetic data may not reflect real patterns
3. Missing clinical features (severity scores, lab values, vital signs)
4. Temporal patterns not captured (discharge timing, follow-up scheduling)

---

## Findings & Insights

### What We Learned

1. **Data Quality:** Excellent
   - Clean synthetic data with proper relationships
   - No missing values or outliers
   - Realistic distributions

2. **Feature Engineering:** Complete but Limited
   - Successfully created 29 features from raw data
   - Features cover demographics, comorbidities, medications, history
   - May not capture key readmission drivers

3. **Model Training:** Successful
   - All 3 models trained without errors
   - MLflow tracking operational
   - Baseline established

4. **Feature Importance Patterns:**
   - Random Forest identifies key predictors
   - Admissions history appears relevant
   - Demographics less predictive than expected

5. **Class Imbalance:** Handled
   - 15.2% readmission rate is imbalanced (84.8% vs 15.2%)
   - Used `class_weight='balanced'` in all models
   - Precision-Recall tradeoff managed

### Technical Achievements

✅ **Data Engineering**
- Containerized infrastructure with Docker
- Airflow DAG orchestration
- PostgreSQL data management
- Automated data validation

✅ **Feature Engineering**
- Systematic feature creation
- Category binning and encoding
- Domain-specific features
- Feature documentation

✅ **ML Lifecycle Management**
- MLflow experiment tracking
- Multiple model training
- Metric logging and comparison
- Reproducible pipeline

✅ **Code Quality**
- Professional code structure
- Error handling
- Logging and monitoring
- Documentation

---

## Future Improvements

### Short Term (Next Iterations)

#### 1. Enhanced Feature Engineering
- **Interaction features:** age × comorbidities, LOS × medications
- **Risk scores:** Combine multiple features into clinical scores
- **Temporal patterns:** Admission frequency trends, seasonal effects
- **Feature importance analysis:** Keep only top predictive features

#### 2. Hyperparameter Tuning
- **Grid search:** Test parameter ranges for each model
- **Cross-validation:** Better performance estimation
- **Threshold tuning:** Optimize for recall vs precision
- **Cost-sensitive learning:** Weight readmission errors heavily

#### 3. Advanced Techniques
- **SMOTE:** Synthetic over-sampling for minority class
- **Ensemble stacking:** Combine model predictions
- **Deep learning:** Neural networks for complex patterns
- **XGBoost/LightGBM:** More advanced boosting

#### 4. Model Evaluation
- **Confusion matrix analysis:** FP vs FN tradeoff
- **Calibration plots:** Probability calibration
- **SHAP values:** Feature importance explanation
- **Learning curves:** Diagnose bias vs variance

### Medium Term (Production Ready)

#### 5. Model Deployment
- **Model serving:** Flask/FastAPI endpoints
- **Real-time prediction:** Integrate with hospital systems
- **Model monitoring:** Detect performance degradation
- **A/B testing:** Compare models in production

#### 6. Data Enhancement
- **Real hospital data:** Replace synthetic with production data
- **Additional features:** Lab values, vital signs, medications
- **Clinical context:** Comorbidity indices, disease severity
- **Social factors:** Transportation, housing, social support

#### 7. Clinical Validation
- **Clinical review:** Validate with domain experts
- **Fairness analysis:** Check for bias across demographics
- **Explainability:** Interpret model decisions
- **Regulatory compliance:** HIPAA, GDPR considerations

#### 8. Operational Excellence
- **Monitoring dashboards:** Track model performance
- **Alert systems:** Notify of data quality issues
- **Documentation:** API documentation, runbooks
- **Maintenance:** Regular retraining, model updates

### Long Term (Enterprise Scale)

#### 9. Advanced Analytics
- **Causal inference:** Understand true drivers of readmission
- **Counterfactual analysis:** "What if" scenarios
- **Time series modeling:** Predict readmission trajectory
- **Clustering:** Patient subgroup identification

#### 10. Integration
- **EHR integration:** Connect to hospital systems
- **Workflow automation:** Automatic alerts to care teams
- **Clinical decision support:** Integrate with provider tools
- **Quality improvement:** Track intervention outcomes

---

## Technical Stack

### Data Engineering
- **Python 3.10** - Primary language
- **Pandas 2.1.0** - Data manipulation
- **NumPy** - Numerical computing
- **PostgreSQL 13** - Data warehouse
- **Apache Airflow 2.7.0** - Workflow orchestration
- **psycopg2** - Python-PostgreSQL adapter

### Machine Learning
- **scikit-learn** - ML algorithms
- **MLflow 2.7.0** - Experiment tracking
- **Jupyter** - Exploratory analysis
- **Matplotlib/Seaborn** - Visualization

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **Redis 7.2** - Message broker
- **Kubernetes (future)** - Orchestration

### Version Control & CI/CD
- **Git/GitHub** - Version control
- **Professional commit messages** - Code history

### Deployment & Monitoring
- **Linux/Ubuntu** - Operating system
- **Port 5432** - PostgreSQL
- **Port 8080** - Airflow
- **Port 5000** - MLflow

---

## How to Run

### Prerequisites
- Docker Desktop installed
- Python 3.10+ with virtual environment
- Required packages: `pip install -r requirements.txt`

### Steps

**1. Start Docker containers:**
```bash
docker-compose up -d
```

**2. Generate and load data:**
```bash
python src/ingestion/generate_data.py
python src/ingestion/validate_data.py
python src/ingestion/load_data.py
```

Or trigger through Airflow:
- Go to http://localhost:8080
- Login: admin/admin
- Trigger `patient_data_pipeline` DAG

**3. Run exploratory analysis:**
```bash
jupyter notebook hospital_eda.ipynb
```

**4. Engineer features:**
```bash
python src/features/feature_engineering.py
```

**5. Train models:**
```bash
python src/modeling/train_model.py
```

**6. View results:**
- Airflow: http://localhost:8080
- MLflow: http://localhost:5000

---

## Conclusion

This project demonstrates a **production-grade end-to-end ML system** with:
- ✅ Containerized infrastructure
- ✅ Automated data pipeline
- ✅ Comprehensive feature engineering
- ✅ Multiple model training
- ✅ Experiment tracking with MLflow
- ✅ Professional code quality
- ✅ Thorough documentation

**Baseline models established with ROC-AUC of 0.5781**, providing foundation for iterative improvement through enhanced feature engineering, hyperparameter tuning, and incorporation of additional clinical data.

This portfolio piece showcases **data engineering skills, ML fundamentals, and professional software development practices** valuable for data engineering and ML engineering roles.

---

**Version:** 1.0 - Baseline Documentation
**Date:** December 25, 2025
**Author:** Bob (Data Engineer)
**Status:** Complete - Ready for Portfolio Showcase

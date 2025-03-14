# New York City Taxi Fare Prediction

## Overview
This project aims to predict taxi fare amounts in New York City using machine learning techniques. The dataset includes taxi trip records, such as pickup/drop-off locations, timestamps, passenger counts, trip distances, and fare amounts. 

## Objectives
- Predict taxi fare amounts using regression models.
- Classify trips into predefined fare categories.
- Analyze ride patterns across geography and time.
- Perform feature engineering to improve predictive performance.
- Compare different machine learning models and evaluate their effectiveness.

## Dataset
- **Source:** [Kaggle - NYC Taxi Trips 2019](https://www.kaggle.com/datasets/dhruvildave/new-york-city-taxi-trips-2019/data)
- **Features:**
  - `tpep_pickup_datetime`, `tpep_dropoff_datetime` (timestamp features)
  - `trip_distance`, `trip_duration`
  - `fare_amount`, `passenger_count`
  - `PULocationID`, `DOLocationID`, `RateCodeID`, `payment_type`
  
## Methodology
### Data Preprocessing
- Handle missing values and filter out invalid data.
- Parse datetime fields to extract temporal features.
- Normalize and standardize numerical features.
- Encode categorical variables for model training.

### Exploratory Data Analysis (EDA)
- Visualize trip distributions using histograms and boxplots.
- Identify correlations using heatmaps and scatter plots.
- Perform dimensionality reduction (PCA, UMAP) to uncover patterns.

### Feature Engineering
- Extract features like `trip_duration`, `average_speed`, `hour_of_day`, `day_of_week`.
- Categorize pickup times into `Morning`, `Afternoon`, `Evening`, and `Night`.
- One-hot encode pickup/drop-off boroughs.
- Create binary features for Manhattan and airport locations.

### Model Selection
- **Regression:** Linear Regression, Random Forest, Gradient Boosting.
- **Classification:** Logistic Regression, Decision Trees.
- **Clustering:** K-Means, DBSCAN for location-based segmentation.
- **Deep Learning:** Neural network model using TensorFlow/PyTorch.

### Model Evaluation
- Compare performance using RMSE, MAE, and R² scores.
- Perform hyperparameter tuning with GridSearchCV.
- Implement cross-validation for robust evaluation.

## Getting Started
### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries (see `requirements.txt`)

### Installation
```bash
# Clone the repository
git clone https://github.com/moira56/Data-Science.git

# Navigate to the project directory
cd Data-Science

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For macOS/Linux
# venv\Scripts\activate  # For Windows (uncomment this line)

# Install dependencies
pip install -r requirements.txt


# Ensure necessary directories exist
mkdir -p taxi_data/2019
mkdir -p taxi_data/taxi_zones

# Move to the data directory
cd taxi_data

# Check if the taxi zone lookup file exists
if [ ! -f "taxi_zone_lookup.csv" ]; then
    echo "Warning: 'taxi_zone_lookup.csv' not found! Make sure to place it in the taxi_data directory."
fi

# Move back to the main project directory
cd ..

### Running Notebooks
jupyter notebook

### Running Scripts
python Taxi_dataset_part1.py
```
### Contributors
Data Scientists: Model development and feature engineering.
Data Engineers: Data cleaning and preprocessing.
Analysts: EDA, visualization, and statistical analysis

###  Deliverables
Mid-Journey Report: Data analysis and preprocessing results.
Final Report: Comprehensive documentation with models and insights.
Presentation: Summary of findings and recommendations.

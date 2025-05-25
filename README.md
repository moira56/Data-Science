
New York City Taxi Trips Fare Prediction and Classification
Project Overview
This repository contains the implementation of machine learning and deep learning models to analyze the New York City Taxi Trips dataset (2019) with the goal of predicting taxi fare amounts (regression) and classifying trips into fare ranges (classification). The project covers the final three phases of the data analytics lifecycle: model building, model evaluation, and operationalization.
________________________________________
Dataset
The dataset used in this project is sourced from Kaggle:
New York City Taxi Trips 2019
https://www.kaggle.com/datasets/dhruvildave/new-york-city-taxi-trips-2019/data
The dataset includes extensive trip data capturing variables such as trip distance, pickup/dropoff times, passenger count, and fare amounts.

Project Goals
1.	Regression Task: Predict the exact fare amount for a taxi trip using trip features.
2.	Classification Task: Classify trips into predefined fare ranges:
o	Class 1: Short trips, low fare (< $10)
o	Class 2: Medium-distance trips, moderate fare ($10 - $30)
o	Class 3: Long-distance trips, high fare ($30 - $60)
o	Class 4: Premium fares (> $60)

Project Structure
1. Data Exploration & Preprocessing
•	Conducted data cleaning, handling missing values, and feature engineering.
•	Applied relevant transformations to improve model performance.
2. Model Building
•	k-Nearest Neighbors (kNN): Implemented from scratch using NumPy arrays.
•	Supervised Learning Models: Tested at least two models from scikit-learn (e.g., Random Forest, Gradient Boosting).
•	Ensemble Models: Applied bagging (Random Forest) and boosting (Gradient Boosting) methods.
•	Deep Learning Model: Developed using TensorFlow/PyTorch (layer-wise implementation or transfer learning).
•	Clustering: Performed clustering with different algorithms (e.g., K-Means, DBSCAN) to explore data patterns.
3. Model Evaluation & Comparison
•	Evaluated models with appropriate metrics (RMSE, MAE, R² for regression; accuracy, precision, recall, F1 for classification).
•	Compared models in tabular format.
•	Discussed strengths, weaknesses, and insights.
4. Operationalization
•	Documented deployment strategies and production environment considerations.
•	Saved trained models for future use.
________________________________________
Deliverables
•	Final Report Comprehensive documentation of all phases, including code snippets, visualizations, model evaluation, and lessons learned.
•	Output: Contain Result & Graphs 
•	Codebase: Python 
________________________________________
Usage Instructions
1.	Install dependencies:
Use the provided requirements.txt file:
pip install -r requirements.txt
2.	Run the notebooks:
o	Start with TaxiAnalysis.ipynb to explore and prepare data..
o	Follow instructions in notebooks for detailed steps and explanations.
3.	Trained models are available in the /models folder for quick inference or further analysis.

________________________________________
Tools & Libraries
•	Python 3.x
•	NumPy
•	pandas
•	scikit-learn
•	TensorFlow / PyTorch
•	Matplotlib / Seaborn
•	Jupyter Notebook
•	LaTeX (for report preparation)
________________________________________


Notes
•	All reports are prepared in LaTeX using IEEE style 
•	The repository is private; please request access if needed.
•	Each team member is familiar with all code and implementation details for the final defense.




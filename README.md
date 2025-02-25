# New York City Taxi Fare Prediction

This project is part of the Data Science Working Group at Code for San Francisco. Other DSWG projects can be found at the [main GitHub repo](https://github.com/sfbrigade/data-science-wg).

#### -- Project Status: Active

## Project Intro/Objective

The purpose of this project is to analyze and model taxi trip data from New York City to predict fare amounts based on various trip characteristics. This analysis aims to provide insights into fare structures and assist in optimizing pricing strategies.

### Methods Used

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Modeling
- Regression Analysis
- Data Visualization

### Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Project Description

In this project, we utilize the [New York City Taxi Trips 2019 dataset](https://www.kaggle.com/datasets/dhruvildave/new-york-city-taxi-trips-2019/data) to develop predictive models for taxi fares. The dataset includes detailed information on taxi trips, such as pickup and drop-off locations, trip distances, passenger counts, and fare amounts.

Key steps in the project include:

1. **Data Cleaning:** Handling missing values, removing duplicates, and correcting data types.
2. **Exploratory Data Analysis:** Visualizing data distributions and relationships between variables to identify patterns and outliers.
3. **Feature Engineering:** Creating new features like trip duration, distance calculations, and time-based features (e.g., hour of day, day of week).
4. **Modeling:** Applying regression techniques to predict fare amounts and evaluating model performance using metrics like RMSE.
5. **Validation:** Testing the model on unseen data to assess its generalization capability.

## Needs of this Project

- Data Scientists with experience in regression modeling.
- Data Engineers skilled in data cleaning and preprocessing.
- Visualization experts to create insightful plots and dashboards.
- Project Managers to coordinate tasks and timelines.

## Getting Started

1. **Clone the repository:**  
   `git clone https://github.com/moira56/Data-Science.git`

2. **Navigate to the project directory:**  
   `cd Data-Science`

3. **Create and activate a virtual environment (optional but recommended):**  
   `python -m venv venv`  
   `source venv/bin/activate`  *(On macOS/Linux)*  
   `venv\Scripts\activate`  *(On Windows)*  

4. **Install required dependencies:**  
   `pip install -r requirements.txt`

5. **Download the dataset:**  
   - The dataset can be accessed from [Kaggle](https://www.kaggle.com/datasets/dhruvildave/new-york-city-taxi-trips-2019/data).  
   - Place the dataset in the `data/` directory.

6. **Run the exploratory data analysis notebook:**  
   `jupyter notebook`  
   - Open `notebooks/EDA.ipynb` and execute all cells to explore the dataset.

7. **Train models and evaluate performance:**  
   - Open and run `notebooks/modeling.ipynb` to train and compare different machine learning models.

8. **Generate final report and results:**  
   - Review `reports/final_report.pdf` for findings, conclusions, and model performance.

9. **Deactivate the virtual environment (if used):**  
   `deactivate`

## Repository Structure

/project-directory │── data/ # Raw and processed dataset files
│── notebooks/ # Jupyter notebooks for data analysis and modeling
│── models/ # Trained models
│── src/ # Scripts for data processing and model training
│── reports/ # Documentation and reports
│── README.md # Project overview and instructions

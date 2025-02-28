"""
Project: NYC Taxi Fare Prediction

Phase 1: Problem Formulation

1. Problem Definition:
The dataset consists of millions of New York City taxi ride records, capturing essential trip details such as pickup and drop-off times, trip distance, fare amount, passenger count, and payment type. The primary objective is to analyze and predict taxi fares based on trip characteristics.

Key research questions:
- Can taxi fares be accurately predicted based on ride attributes?
- What factors (e.g., time of day, trip distance, payment type) have the most significant impact on fare price?
- Can taxi trips be classified into predefined fare categories (low, medium, high, premium) for better pricing strategies?

2. Goals and Objectives:
- Develop regression models to predict taxi fare amounts
- Implement classification models to categorize trips based on fare price ranges
- Conduct exploratory data analysis (EDA) to identify patterns and trends within the dataset
- Analyze the influence of different factors on fare price determination
- Provide insights that could benefit passengers, taxi companies, and policymakers in optimizing fare structures and taxi availability

Phase 2: Data Analysis and Cleansing

1. Dataset Description:
The dataset used in this project is the New York City Taxi Trip Dataset, obtained from Kaggle.
It contains detailed trip records from taxi services in New York City, including information
such as pickup and drop-off locations, timestamps, trip distance, fare amount, and payment type.

2. Source:
The dataset is publicly available at:
https://www.kaggle.com/datasets/dhruvildave/new-york-city-taxi-trips-2019/data

3. Data Fields:
- Trip details: pickup and drop-off timestamps, trip distance, passenger count
- Fare information: base fare, total amount, additional surcharges, and payment type
- Location data: pickup and drop-off zone identifiers

4. Objective of Data Pre-processing:
- Handle missing or inconsistent data entries
- Remove outliers and invalid values
- Normalize numerical data for model training
"""
import sqlite3
import pandas as pd

# Lista baza podataka
database_files = [
    r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-01.sqlite",
    r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-02.sqlite",
    r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-03.sqlite", 
     r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-04.sqlite",
      r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-05.sqlite",
       r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-06.sqlite",
        r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-07.sqlite",
         r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-08.sqlite",
          r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-09.sqlite",
             r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-10.sqlite",
                r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-11.sqlite",
                   r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\data\2019\2019-12.sqlite",
    # Ovdje dodajte ostale baze
]

dataframes = []

# Učitavanje podataka po manjim dijelovima
for db_file in database_files:
    try:
        # Otvaranje konekcije s bazom
        conn = sqlite3.connect(db_file)
        query = "SELECT * FROM tripdata LIMIT 100000"  # Učitavanje samo prvih 100000 redova
        df_chunk = pd.read_sql_query(query, conn)
        dataframes.append(df_chunk)
        conn.close()
        print(f"Podaci učitani iz {db_file}")
    except Exception as e:
        print(f"Greška pri učitavanju podataka iz {db_file}: {e}")

# Kombiniranje svih manjih dijelova u jedan veliki DataFrame
df_all = pd.concat(dataframes, ignore_index=True)

# Uklanjanje dupliciranih redova
df_all.drop_duplicates(inplace=True)

# Uštediti podatke u CSV datoteku
df_all.to_csv("combined_data.csv", index=False)
print("Svi podaci su spojeni u combined_data.csv.")

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Izdvajanje samo numeričkih kolona za normalizaciju i standardizaciju
numerical_cols = df_all.select_dtypes(include=['float64', 'int64']).columns

# Kreiranje MinMaxScaler objekta za normalizaciju
scaler = MinMaxScaler()

# Normalizacija podataka
df_all_normalized = pd.DataFrame(scaler.fit_transform(df_all[numerical_cols]), columns=numerical_cols)

# Zamjena originalnih numeričkih kolona s normaliziranim podacima
df_all[numerical_cols] = df_all_normalized

# Spremanje normaliziranih podataka u CSV datoteku
df_all.to_csv("normalized_data.csv", index=False)
print("Normalizirani podaci spremljeni u 'normalized_data.csv'.")

# Kreiranje StandardScaler objekta za standardizaciju
standard_scaler = StandardScaler()

# Standardizacija podataka
df_standardized = pd.DataFrame(standard_scaler.fit_transform(df_all[numerical_cols]), columns=numerical_cols)

# Zamjena originalnih numeričkih kolona s standardiziranim podacima
df_all[numerical_cols] = df_standardized

# Spremanje standardiziranih podataka u CSV datoteku
df_all.to_csv("standardized_data.csv", index=False)
print("Standardizirani podaci spremljeni u 'standardized_data.csv'.")

import os
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, levene
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Set global styling
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({"figure.dpi": 120})

#%%
# Step 1: Data Loading and Precproceesing
import pandas as pd
import sqlite3
import numpy as np
import os

def load_and_sample(data_dir, sample_frac=0.15):
    all_dfs = []
    for file in os.listdir(data_dir):
        if file.endswith(".sqlite"):
            conn = sqlite3.connect(os.path.join(data_dir, file))
            chunksize = 100_000  # Process 100k rows at a time

            # Read in chunks to avoid memory overflow
            for df_chunk in pd.read_sql(
                "SELECT * FROM tripdata",
                conn,
                chunksize=chunksize,
                parse_dates=['tpep_pickup_datetime']
            ):
                # Preprocessing steps
                df_chunk['pickup_hour'] = df_chunk['tpep_pickup_datetime'].dt.hour

                # Create fare_class
                bins = [-np.inf, 10, 30, 60, np.inf]
                labels = ["Class 1", "Class 2", "Class 3", "Class 4"]
                df_chunk['fare_class'] = pd.cut(
                    df_chunk['fare_amount'],
                    bins=bins,
                    labels=labels,
                    right=True,
                    include_lowest=True
                )

                # Stratified sampling in one step
                # Stratify by both fare_class and pickup_hour
                stratified = df_chunk.groupby(
                    by=['fare_class', 'pickup_hour'],
                    observed=True  # Silence future warning
                ).apply(
                    lambda x: x.sample(frac=sample_frac, random_state=42,
                                      ignore_index=False),
                    include_groups=False  # Silence deprecation warning
                ).reset_index(drop=True)

                all_dfs.append(stratified)
            conn.close()

    # Concatenate all chunks and reset index
    return pd.concat(
    [df for df in all_dfs if not df.empty and not df.isna().all().all()],
    ignore_index=True
)


# Taking 10% of data as sample 
df = load_and_sample(r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\taxi_data\2019", sample_frac=0.1)


df.describe()

#%%
import pandas as pd
import numpy as np

# --- Data Cleaning ---

# 1. Drop duplicates
df.drop_duplicates(inplace=True)

# 2. Handle missing values
# Drop rows where critical columns have missing values
critical_columns = [
    "fare_amount",
    "trip_distance",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "pulocationid",  # Pickup location ID
    "dolocationid"   # Dropoff location ID
]
df.dropna(subset=critical_columns, inplace=True)

# 3. Convert datetime columns
# Ensure pickup and dropoff datetimes are valid
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

# Drop rows with invalid datetime values
df = df[
    (df["tpep_pickup_datetime"].notnull()) &
    (df["tpep_dropoff_datetime"].notnull())
]

# 4. Calculate trip duration (in minutes)
df["trip_duration"] = (
    (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
)

# 5. Remove outliers
# - Filter trips within 2019 (dataset is for 2019)
df = df[
    (df["tpep_pickup_datetime"].dt.year == 2019) &
    (df["tpep_dropoff_datetime"].dt.year == 2019)
]

# - Valid trip distance (0 < distance <= 100 miles)
# Adjusted based on `df.describe()` showing a max of ~311 miles
df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]

# - Valid fare amount (0 < fare <= $500)
# Adjusted based on `df.describe()` showing a max of ~$411k (likely erroneous)
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 500)]

# - Valid passenger count (1 <= passengers <= 6)
df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

# - Valid trip duration (0 < duration <= 1440 minutes (24 hours))
df = df[(df["trip_duration"] > 0) & (df["trip_duration"] <= 1440)]

# - Remove trips with negative or unrealistic tolls/tips
df = df[
    (df["tip_amount"] >= 0) &
    (df["tolls_amount"] >= 0) &
    (df["congestion_surcharge"] >= 0)
]

# 6. Encode `store_and_fwd_flag`
df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y": 1, "N": 0})

# 7. Create fare classes for classification
bins = [-np.inf, 10, 30, 60, np.inf]
labels = ["Class 1", "Class 2", "Class 3", "Class 4"]
df["fare_class"] = pd.cut(df["fare_amount"], bins=bins, labels=labels)

# 8. Reset index
df.reset_index(drop=True, inplace=True)

df.head()
df.describe()

# %%
# Step 2: Exploratory Data Analysis (EDA)

# --- Descriptive Statistics ---

print("Descriptive Statistics:")
numerical_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
print(df[numerical_features].describe())


# --- Visualizations ---

# 1. Distribution of All Numerical Features
def plot_feature_distributions(dataframe, numerical_cols):
    n_cols = 3  # Number of subplots per row
    n_rows = (len(numerical_cols) // n_cols) + (len(numerical_cols) % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # Flatten axes array for easy iteration

    for i, col in enumerate(numerical_cols):
        sns.histplot(dataframe[col], bins=50, kde=True, ax=axes[i], color="teal")
        axes[i].set_title(f"Distribution of {col}", pad=10, weight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    # Hide unused subplots
    for j in range(len(numerical_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

plot_feature_distributions(df, numerical_features)

# 2. Correlation Heatmap for All Numerical Features
plt.figure(figsize=(20, 18))
corr_matrix = df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, annot_kws={"size": 10}, linewidths=0.5)
plt.title("Correlation Matrix for All Numerical Features", pad=20, weight="bold")
plt.tight_layout()
plt.show()

# 3. Boxplots for All Numerical Features
def plot_boxplots(dataframe, numerical_cols):
    n_cols = 3  # Number of subplots per row
    n_rows = (len(numerical_cols) // n_cols) + (len(numerical_cols) % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # Flatten axes array for easy iteration

    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=dataframe[col], ax=axes[i], color="gold", width=0.5, fliersize=3)
        axes[i].set_title(f"Boxplot of {col}", pad=10, weight="bold")
        axes[i].set_xlabel(col)
        axes[i].grid(axis="x", linestyle="--", alpha=0.7)

    # Hide unused subplots
    for j in range(len(numerical_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

plot_boxplots(df, numerical_features)

# 4. Time Series: Average Fare by Hour (if applicable)
if "tpep_pickup_datetime" in df.columns:
    df["pickup_hour"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.hour
    hourly_fare = df.groupby("pickup_hour")["fare_amount"].mean()
    plt.figure(figsize=(14, 7))
    sns.lineplot(x=hourly_fare.index, y=hourly_fare.values, marker="o", markersize=8, linewidth=2.5, color="purple")
    plt.title("Average Fare Amount by Pickup Hour", pad=20, weight="bold")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Fare ($)")
    plt.grid(linestyle="--", alpha=0.7)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.show()

# 5. Fare Class Distribution (if applicable)
if "fare_class" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(x="fare_class", data=df, palette="Set2", order=["Class 1", "Class 2", "Class 3", "Class 4"])
    plt.title("Distribution of Fare Classes", pad=20, weight="bold")
    plt.xlabel("Fare Class")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

#%%
# Step 3: Dimensionality Reduction

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

# # Set NUMBA_NUM_THREADS first (before any imports)
# os.environ["NUMBA_NUM_THREADS"] = "8"  # Critical: Set before imports

# --- Feature Engineering ---
# 1. Extract hour from pickup datetime
df["hour"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.hour.astype("Int8")

# 2. Calculate average speed (miles per hour)
df["speed"] = df["trip_distance"] / (df["trip_duration"] / 60)  # miles per hour
df["speed"] = df["speed"].replace([np.inf, -np.inf], np.nan).astype("float32")  # Handle division by zero

# 3. One-hot encode Manhattan pickups and dropoffs
manhattan_ids = [4, 12, 13, 43, 45, 79, 87, 88, 90, 100, 113, 114, 140, 141, 142, 
                 143, 151, 161, 162, 163, 164, 239, 246, 249, 261, 262, 263]  # IDs for Manhattan zones

df["PU_B_Manhattan"] = df["pulocationid"].isin(manhattan_ids).astype("Int8")
df["DO_B_Manhattan"] = df["dolocationid"].isin(manhattan_ids).astype("Int8")

# --- Features for Dimensionality Reduction ---
features = [
    "trip_distance",          # Distance traveled
    "fare_amount",            # Target variable (for visualization)
    "trip_duration",          # Trip duration in minutes
    "passenger_count",        # Number of passengers
    "hour",                   # Hour of pickup (engineered feature)
    "PU_B_Manhattan",         # Pickup in Manhattan (one-hot encoded)
    "DO_B_Manhattan",         # Dropoff in Manhattan (one-hot encoded)
    "speed"                   # Average speed (engineered feature)
]

# Ensure features are numeric and handle missing values
X = df[features].select_dtypes(include=["number"]).dropna()

# Reduce dataset size for faster computation
sample_size = 100_000  # Adjust this based on your system's memory and speed requirements
X_sample = X.sample(n=sample_size, random_state=42)

# --- Preprocessing ---
# Use float32 to reduce memory usage
X_sample = X_sample.astype("float32")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# --- PCA ---
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_scaled)

# --- UMAP ---
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,           # Smaller n_neighbors for faster computation
    min_dist=0.1,             # Smaller min_dist for faster computation
    metric="euclidean",
    n_jobs=-1,                # Use all available cores
    low_memory=True           # Optimize for memory usage
)
umap_result = umap_model.fit_transform(X_scaled)

# --- Plotting (memory-efficient) ---
plt.figure(figsize=(18, 6))

# PCA Plot
plt.subplot(1, 2, 1)
plt.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    c=df.loc[X_sample.index, "fare_amount"],  # Align fare_amount with scaled data
    cmap="viridis",
    alpha=0.3,
    s=2  # Smaller points for memory efficiency
)
plt.colorbar(label="Fare Amount ($)")
plt.title("PCA Result (Colored by Fare)", fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# UMAP Plot
plt.subplot(1, 2, 2)
plt.scatter(
    umap_result[:, 0],
    umap_result[:, 1],
    c=df.loc[X_sample.index, "fare_amount"],  # Align fare_amount with scaled data
    cmap="viridis",
    alpha=0.3,
    s=2
)
plt.colorbar(label="Fare Amount ($)")
plt.title("UMAP Result (Colored by Fare)", fontweight="bold")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")

plt.tight_layout()
plt.show()

#%%
# Step 4: Hypothesis Testing

# --- Hypothesis Testing ---
# Hypothesis: Trip distance affects fare amount
short_trips = df[df["trip_distance"] <= 5]["fare_amount"]
long_trips = df[df["trip_distance"] > 5]["fare_amount"]

# Perform t-test
t_stat, p_value = ttest_ind(short_trips, long_trips)
print(f"\nHypothesis Test: Trip Distance vs Fare Amount")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Perform Levene's test for equal variances
levene_stat, levene_p_value = levene(short_trips, long_trips)
print(f"Levene's Test for Equal Variances")
print(f"Levene Statistic: {levene_stat:.4f}, P-value: {levene_p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis: Trip distance significantly affects fare amount.")
else:
    print("Fail to reject the null hypothesis.")

# Visualize distributions of fare_amount for short and long trips
plt.figure(figsize=(14, 6))
sns.histplot(short_trips, bins=50, kde=True, label="Short Trips (<= 5 miles)", color="blue", alpha=0.5)
sns.histplot(long_trips, bins=50, kde=True, label="Long Trips (> 5 miles)", color="orange", alpha=0.5)
plt.title("Distribution of Fare Amount by Trip Distance", weight="bold")
plt.xlabel("Fare Amount ($)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# Step 5: Feature Engineering

import pandas as pd
import os

# Set the correct path for the CSV file
csv_path = r"C:\Users\mceka\Desktop\Wa-vjezbe\Data-Science\taxi_data\taxi_zone_lookup.csv"

# Check if the file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

# Load the CSV file
zone_lookup = pd.read_csv(csv_path)

# --- Feature Engineering ---
# --- Time-Based Features ---
df["pickup_day"] = df["tpep_pickup_datetime"].dt.dayofweek
df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

# Binning pickup_hour into categories
df["pickup_time_of_day"] = pd.cut(
    df["pickup_hour"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"]
)

# --- Speed Feature ---
df["average_speed"] = df["trip_distance"] / (df["trip_duration"] / 60)  # miles per hour

# --- Categorical Encoding ---
# Encode RateCodeID and Payment_type (numeric codes)
df["RateCodeID"] = df["ratecodeid"].astype("category").cat.codes
df["payment_type"] = df["payment_type"].astype("category").cat.codes

# --- Location Mapping ---
zone_dict = dict(zip(zone_lookup["LocationID"], zone_lookup["Zone"]))
borough_dict = dict(zip(zone_lookup["LocationID"], zone_lookup["Borough"]))

df["PULocationID"] = df["pulocationid"].map(zone_dict).fillna("Unknown")
df["DOLocationID"] = df["dolocationid"].map(zone_dict).fillna("Unknown")
df["PU_borough"] = df["pulocationid"].map(borough_dict).fillna("Unknown")
df["DO_borough"] = df["dolocationid"].map(borough_dict).fillna("Unknown")

# One-hot encode pickup/dropoff boroughs
df = pd.get_dummies(df, columns=["PU_borough", "DO_borough"], prefix=["PU_B", "DO_B"], drop_first=True)

# --- Additional Features ---
# Distance to/from Manhattan
manhattan_id = zone_lookup[zone_lookup["Borough"] == "Manhattan"]["LocationID"].values
df["is_pickup_manhattan"] = df["pulocationid"].isin(manhattan_id).astype(int)
df["is_dropoff_manhattan"] = df["dolocationid"].isin(manhattan_id).astype(int)

# Distance to/from Airports
airport_ids = zone_lookup[zone_lookup["service_zone"] == "Airports"]["LocationID"].values
df["is_pickup_airport"] = df["pulocationid"].isin(airport_ids).astype(int)
df["is_dropoff_airport"] = df["dolocationid"].isin(airport_ids).astype(int)

# --- Final Feature List ---
print("\nFinal Features:")
print(df.columns.tolist())

print(*df.columns.tolist(), sep='\n')

#%%
# --- Model Selection ---

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Define Features and Targets ---

# Features for regression and classification tasks
X_reg = df[[
    "trip_distance",          # Distance traveled
    "pickup_hour",                   # Hour of pickup (engineered feature)
    "average_speed",                  # Average speed during the trip (renamed from `average_speed`)
    "RateCodeID",             # Encoded rate code
    "payment_type",           # Encoded payment type
    "is_pickup_manhattan",    # Pickup in Manhattan
    "is_dropoff_manhattan",   # Dropoff in Manhattan
    "is_pickup_airport",      # Pickup at airport
    "is_dropoff_airport",     # Dropoff at airport
    "PU_B_Brooklyn",          # Pickup borough: Brooklyn
    "PU_B_Manhattan",         # Pickup borough: Manhattan
    "PU_B_Queens",            # Pickup borough: Queens
    "PU_B_Staten Island",     # Pickup borough: Staten Island
    "PU_B_Unknown",           # Pickup borough: Unknown
    "DO_B_Brooklyn",          # Dropoff borough: Brooklyn
    "DO_B_Manhattan",         # Dropoff borough: Manhattan
    "DO_B_Queens",            # Dropoff borough: Queens
    "DO_B_Staten Island",     # Dropoff borough: Staten Island
    "DO_B_Unknown"            # Dropoff borough: Unknown
]]

# Regression target: `fare_amount`
y_reg = df["fare_amount"]

# Classification target: `fare_class` (predefined fare ranges)
y_clf = df["fare_class"]

# --- Split Data into Training and Testing Sets ---
from sklearn.model_selection import train_test_split

# Regression split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Classification split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_reg, y_clf, test_size=0.2, random_state=42
)

# --- Regression Models ---
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

regression_models = {
    "Linear Regression": LinearRegression(),  # Baseline regression model
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),  # Non-linear model
    #"Random Forest Regressor": RandomForestRegressor(random_state=42)  # Ensemble model
}

# --- Classification Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# classification_models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),  # Baseline classification model
#     "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),  # Non-linear model
#     #"Random Forest Classifier": RandomForestClassifier(random_state=42)  # Ensemble model
# }
# --- Classification Models with Scaling ---
classification_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000))  # Increased max_iter
    ]),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
    # Add RandomForestClassifier here if needed
}

# --- Validation Method: K-Fold Cross-Validation ---
print("Regression Model Evaluation:")
for name, model in regression_models.items():
    scores = cross_val_score(model, X_train_reg, y_train_reg, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-scores)
    print(f"{name} Cross-Validation RMSE: {rmse_scores.mean():.2f}")

print("\nClassification Model Evaluation:")
for name, model in classification_models.items():
    scores = cross_val_score(model, X_train_clf, y_train_clf, cv=5, scoring="accuracy")
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.2f}")



# --- Justification for Validation Method ---
"""
K-Fold Cross-Validation (k=5) is chosen to:
- Ensure robust performance estimation by averaging results across 5 subsets.
- Avoid overfitting by testing on multiple validation sets.
- Handle the large dataset efficiently without data leakage.
"""

# --- Model Suitability Justification ---
"""
1. **Linear Regression**: Baseline model for regression tasks; assumes linear relationships between features and target.
2. **Decision Tree Regressor/Classifier**: Captures non-linear relationships and interactions between features.
3. **Random Forest Regressor/Classifier**: Improves accuracy through ensemble learning (bagging) and handles overfitting better than individual decision trees.
4. **Logistic Regression**: Baseline for classification tasks; works well for linearly separable classes.
5. **Ensemble Methods**: Random Forest is chosen for its ability to generalize well and reduce variance compared to individual decision trees.
"""



import pandas as pd

# Load dataset
df = pd.read_csv("data/citynet_churn.csv")

# 1. Dataset dimensions
print("\n===== DATASET SHAPE =====")
print(df.shape)

# 2. Column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# 3. First five records
print("\n===== FIRST 5 ROWS =====")
print(df.head())

# 4. Dataset information
print("\n===== DATASET INFO =====")
df.info()

# 5. Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 6. Target distribution
print("\n===== CHURN DISTRIBUTION =====")
print(df["churned"].value_counts())

# 7. Target percentages
print("\n===== CHURN PERCENTAGES =====")
print(df["churned"].value_counts(normalize=True) * 100)
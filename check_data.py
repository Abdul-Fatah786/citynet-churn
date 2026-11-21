import pandas as pd
import os

file_path = "data/new_customers.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

else:
    print("File not found:", file_path)
    print("\nFiles available in data folder:")
    print(os.listdir("data"))
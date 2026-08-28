import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Insurance.csv")

# Display basic information
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
df.info()
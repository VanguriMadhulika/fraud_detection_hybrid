import pandas as pd

# Try to load the CSV file
try:
    df = pd.read_csv("data/creditcard.csv")
    print("✅ CSV file found and loaded successfully!")
    print("First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("❌ ERROR: File not found at 'data/creditcard.csv'")
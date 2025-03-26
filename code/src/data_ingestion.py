import pandas as pd

def load_transaction_data(file_path):
    """Loads transaction data from a CSV file."""
    file_path = "../data/customer_transactions_10k.csv"
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} transactions.")
    return df

# Load dataset
df = load_transaction_data("customer_transactions_10k.csv")
print(df.head())

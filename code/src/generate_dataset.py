import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 10,000 transactions
num_rows = 10000
customer_ids = np.random.randint(100000, 999999, num_rows)
amounts = np.random.uniform(-5000, 25000, num_rows) # Transaction amounts
balances = np.random.uniform(-1000, 50000, num_rows) # Account balance
transaction_types = np.random.choice(
["Cash Withdrawal", "Online Transfer", "POS Purchase", "Wire Transfer", "Check Deposit"], num_rows
)
withdrawal_frequency = np.random.randint(1, 16, num_rows) # Times withdrawn in a month
hourly_withdrawals = np.random.randint(1, 11, num_rows) # Withdrawals in last 24h
disputed = np.random.choice([True, False], num_rows) # Transaction disputes
geo_deviation = np.random.randint(0, 5001, num_rows) # Distance variation
high_txn_flag = (amounts > 10000).astype(int) # High-value transaction flag
negative_balance_flag = (balances < 0).astype(int) # Negative balance flag
fraudulent_flag = np.random.choice([0, 1], num_rows, p=[0.95, 0.05]) # Simulated fraud labels

# Add Country for regulatory compliance
countries = ["USA", "India", "UK", "Germany", "Russia", "China", "Canada", "Australia", "France", "Japan"]
country_column = np.random.choice(countries, num_rows)

# Create DataFrame with correct column names
df = pd.DataFrame({
"Customer_ID": customer_ids,
"Amount": amounts.round(2), # Matches risk model
"Balance": balances.round(2),
"Transaction_Type": transaction_types, # Matches risk model
"Withdrawal_Frequency": withdrawal_frequency,
"Hourly_Withdrawals": hourly_withdrawals,
"Disputed": disputed,
"Geo_Location_Deviation": geo_deviation,
"High_Transaction_Flag": high_txn_flag,
"Negative_Balance_Flag": negative_balance_flag,
"Fraudulent_Flag": fraudulent_flag,
"Country": country_column # Matches compliance check
})

# Save dataset in the correct location
dataset_path = "../data/customer_transactions_10k.csv"
df.to_csv(dataset_path, index=False)

print(f"✅ Dataset Generated: {dataset_path}")

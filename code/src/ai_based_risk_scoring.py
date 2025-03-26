import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# File paths (ensure these CSVs exist)
transactions_file = "../data/customer_transactions_10k.csv"
rules_file = "../data/genai_regulatory_rules.csv"
output_file = "../data/final_risk_compliance_results.csv"

# Load customer transactions
try:
    transactions_df = pd.read_csv(transactions_file)
except FileNotFoundError:
    print(f"Error: {transactions_file} not found.")
    exit()

# Load AI-generated regulatory rules
try:
    rules_df = pd.read_csv(rules_file)
except FileNotFoundError:
    print(f"Error: {rules_file} not found.")

    rules_df = pd.DataFrame(columns=["Rule_ID", "Regulatory_Rule"])

# Encode categorical variables
le = LabelEncoder()
transactions_df["Transaction_Type"] = le.fit_transform(transactions_df["Transaction_Type"])
transactions_df["Country"] = le.fit_transform(transactions_df["Country"])

# Create a simulated risk label (0 = Low Risk, 1 = High Risk) based on transaction amount
transactions_df["Risk_Label"] = (transactions_df["Amount"] > 5000).astype(int)

# Prepare training data
X = transactions_df[["Amount", "Transaction_Type", "Country"]]
y = transactions_df["Risk_Label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict risk scores
transactions_df["Predicted_Risk_Score"] = model.predict_proba(X)[:, 1] * 10 # Convert probability to 1-10 scale
transactions_df["Predicted_Risk_Score"] = transactions_df["Predicted_Risk_Score"].astype(int)

# Function to check compliance based on regulatory rules
def check_compliance(transaction):
    issues = []

    # Rule: Transactions exceeding $10,000 require AML reporting
    if transaction["Amount"] > 10000:
        issues.append("AML reporting required for high-value transaction")

    # Rule: More than five large withdrawals in 24 hours triggers review
    if transaction["Transaction_Type"] == "Withdrawal" and transaction["Amount"] > 5000:
        issues.append("High withdrawal amount, review required")

    # Rule: Negative balance for 30+ days is non-compliant
    if "Balance" in transaction and transaction["Balance"] < 0:
        issues.append("Account negative balance for 30+ days")

    # Rule: Transactions from high-risk countries require enhanced due diligence
    high_risk_countries = ["Russia", "China"]
    if transaction["Country"] in high_risk_countries:
        issues.append("Transaction from high-risk country")

    # Determine compliance
    compliance_status = "No" if issues else "Yes"
    remediation = "; ".join(issues) if issues else "Compliant"

    return compliance_status, remediation

# Apply compliance rules
transactions_df["Compliant"], transactions_df["Remediation"] = zip(
    *transactions_df.apply(check_compliance, axis=1)
)

# Save results
transactions_df.to_csv(output_file, index=False)

print(f"Risk scoring and compliance checks completed. Results saved to {output_file}.")
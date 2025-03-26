import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# File paths
DATA_PATH = "../data/customer_transactions_10k.csv"
RULES_PATH = "../data/genai_regulatory_rules.csv"
OUTPUT_PATH = "../data/final_risk_compliance_results.csv"

def process_risk_compliance():
    print("🚀 Starting AI-Powered Risk & Compliance Analysis...")

# Load customer transactions
    try:
       transactions_df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    # Load AI-generated regulatory rules
    try:
        rules_df = pd.read_csv(RULES_PATH)
    except FileNotFoundError:
        print(f"⚠️ Warning: {RULES_PATH} not found. Proceeding without it.")
        rules_df = pd.DataFrame(columns=["Rule_ID", "Regulatory_Rule"])

    # Encode categorical variables
    le = LabelEncoder()
    transactions_df["Transaction_Type"] = le.fit_transform(transactions_df["Transaction_Type"])

    # Simulated risk label (0 = Low Risk, 1 = High Risk) based on transaction amount
    transactions_df["Risk_Label"] = (transactions_df["Amount"] > 5000).astype(int)

    # Prepare training data
    X = transactions_df[["Amount", "Transaction_Type"]]
    y = transactions_df["Risk_Label"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict risk scores (convert probability to 1-10 scale)
    transactions_df["Predicted_Risk_Score"] = model.predict_proba(X)[:, 1] * 10
    transactions_df["Predicted_Risk_Score"] = transactions_df["Predicted_Risk_Score"].astype(int)

    # Compliance Check Function
    def check_compliance(transaction):
        issues = []
        if transaction["Amount"] > 10000:
            issues.append("AML reporting required")
        if transaction["Transaction_Type"] == "Withdrawal" and transaction["Amount"] > 5000:
            issues.append("High withdrawal amount flagged")
        if transaction["Balance"] < 0:
            issues.append("Negative balance issue")

        compliance_status = "No" if issues else "Yes"
        remediation = "; ".join(issues) if issues else "Compliant"
        return compliance_status, remediation

    # Apply compliance rules
    transactions_df["Compliant"], transactions_df["Remediation"] = zip(
    *transactions_df.apply(check_compliance, axis=1)
    )

    # Save results
    transactions_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Risk scoring and compliance checks completed. Results saved to {OUTPUT_PATH}.")

if __name__ == "__main__":
    process_risk_compliance()
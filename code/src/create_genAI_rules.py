
import os
import pandas as pd
from transformers import pipeline

# Load an open-source model (No token required)
generator = pipeline(
"text-generation",
     model="mistralai/Mistral-7B-Instruct-v0.1",
     device_map="auto"
)

# Example prompt for generating regulatory rules
prompt = "Generate advanced banking compliance rules based on federal regulations."
response = generator(prompt, max_length=150, num_return_sequences=5)

rules = [r['generated_text'].strip() for r in response]

rules_df = pd.DataFrame({"Rule_ID": range(1, len(rules) + 1), "Regulatory_Rule": rules})
rules_df.to_csv("data/genai_regulatory_rules.csv", index=False)
# Print generated rules
print("\n AI Generated Compliance Rules Saved: data/genai_regulatory_rules.csv")
print("\n Sample Rules:")
print(rules_df.head())
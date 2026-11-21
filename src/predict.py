
import pandas as pd
import joblib
import os
import argparse

# 1. Accept a CSV path from the command line
parser = argparse.ArgumentParser(description="Predict customer churn")
parser.add_argument(
    "csv_path",
    nargs="?",
    default="data/new_customers.csv"
)
args = parser.parse_args()

# 2. Load the final trained model
model = joblib.load("models/linear_svm_pipeline.joblib")

# 3. Load new customers
new_customers = pd.read_csv(args.csv_path)

if "customer_id" not in new_customers.columns:
    raise ValueError("Input CSV must contain customer_id")

print("New customers loaded:", len(new_customers))

# 4. Keep customer IDs separate
customer_ids = new_customers["customer_id"]

# 5. Remove ID before prediction
X_new = new_customers.drop(columns=["customer_id"])

# 6. Generate predictions and decision scores
predictions = model.predict(X_new)
scores = model.decision_function(X_new)

# 7. Create results
results = pd.DataFrame({
    "customer_id": customer_ids,
    "predicted_churn": predictions,
    "svm_score": scores
})

# 8. Convert numeric predictions to readable labels
results["predicted_churn"] = results["predicted_churn"].map({
    0: "No Churn",
    1: "Churn"
})

# 9. Rank customers by risk
results = results.sort_values(
    by="svm_score",
    ascending=False
).reset_index(drop=True)

results["risk_rank"] = range(1, len(results) + 1)

# 10. Save predictions
os.makedirs("reports", exist_ok=True)

results.to_csv(
    "reports/predictions.csv",
    index=False
)

# 11. Display results
print("\n===== CHURN PREDICTIONS =====")
print(results.to_string(index=False))

print("\nPredictions saved successfully!")
print("File: reports/predictions.csv")

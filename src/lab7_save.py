# ==========================================
# CITYNET CHURN - LINEAR SVM MODEL
# ==========================================

import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/citynet_churn.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

target = "churned"

# Exclude the customer ID and possible target leakage column.
excluded_columns = [
    "customer_id",
    "churned",
    "cancel_request_logged"
]

X = df.drop(columns=excluded_columns)
y = df[target]

print("\nFeatures:")
print(X.columns.tolist())


# ==========================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# Reserve 20% of the data for final testing.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split the remaining 80% into 60% training and 20% validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\n===== DATA SPLIT =====")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))


# ==========================================
# 4. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ==========================================

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ==========================================
# 5. PREPROCESSING
# ==========================================

def make_preprocess(scale=True):
    """Create preprocessing for numeric and categorical features."""

    # Numeric columns: fill missing values, optionally scale.
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical columns: fill missing values and one-hot encode.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Apply the appropriate pipeline to each column group.
    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    return preprocess


# ==========================================
# 6. MODEL EVALUATION FUNCTION
# ==========================================

def test_model(name, preprocess, model):
    """Train a pipeline on training data and evaluate on validation data."""

    pipeline = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    start = time.time()

    # Fit only on the training split.
    pipeline.fit(X_train, y_train)

    # Use decision scores to calculate average precision / PR-AUC.
    validation_scores = pipeline.decision_function(X_val)

    pr_auc = average_precision_score(
        y_val,
        validation_scores
    )

    seconds = time.time() - start

    print(
        f"{name}: "
        f"PR-AUC = {pr_auc:.4f}, "
        f"Time = {seconds:.2f} seconds"
    )

    return pipeline, pr_auc, seconds


# ==========================================
# 7. COMPARE MODELS USING VALIDATION DATA
# ==========================================

results = []

# Logistic Regression
logistic_pipeline, logistic_score, logistic_time = test_model(
    "Logistic Regression",
    make_preprocess(scale=True),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
)

results.append({
    "model": "Logistic Regression",
    "pr_auc": logistic_score,
    "seconds": logistic_time
})


# Linear SVM with scaling
svm_scaled, svm_scaled_score, svm_scaled_time = test_model(
    "SVM linear (scaled)",
    make_preprocess(scale=True),
    SVC(
        kernel="linear",
        class_weight="balanced"
    )
)

results.append({
    "model": "SVM linear (scaled)",
    "pr_auc": svm_scaled_score,
    "seconds": svm_scaled_time
})


# Linear SVM without scaling
svm_unscaled, svm_unscaled_score, svm_unscaled_time = test_model(
    "SVM linear (no scaling)",
    make_preprocess(scale=False),
    SVC(
        kernel="linear",
        class_weight="balanced"
    )
)

results.append({
    "model": "SVM linear (no scaling)",
    "pr_auc": svm_unscaled_score,
    "seconds": svm_unscaled_time
})


# ==========================================
# 8. SAVE VALIDATION MODEL COMPARISON
# ==========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df = pd.DataFrame(results).sort_values(
    by="pr_auc",
    ascending=False
)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\n===== VALIDATION MODEL COMPARISON =====")
print(results_df.to_string(index=False))


# ==========================================
# 9. FINAL TEST EVALUATION OF SVM
# ==========================================

# The SVM has not been trained on X_test.
# Evaluate it on the untouched test split before final retraining.
test_scores = svm_scaled.decision_function(X_test)

test_pr_auc = average_precision_score(
    y_test,
    test_scores
)

print("\n===== FINAL TEST EVALUATION =====")
print("Test PR-AUC:", round(test_pr_auc, 4))


# ==========================================
# 10. TEST TOP-100 EVALUATION
# ==========================================

# Select up to 100 customers with the highest test churn scores.
top = np.argsort(-test_scores)[:100]

top_customers = X_test.iloc[top].copy()

# Add IDs and actual labels for offline evaluation.
top_customers["customer_id"] = df.loc[
    X_test.index[top],
    "customer_id"
].values

top_customers["actual_churned"] = y_test.iloc[top].values
top_customers["svm_score"] = test_scores[top]

top_customers.to_csv(
    "reports/test_top_100_call_list.csv",
    index=False
)

churners_in_top100 = int(
    top_customers["actual_churned"].sum()
)

random_expected = y_test.mean() * len(top_customers)

print("\n===== TEST TOP 100 EVALUATION =====")
print("Customers selected:", len(top_customers))
print("Actual churners in top 100:", churners_in_top100)
print(
    "Expected churners in random 100:",
    round(random_expected, 2)
)
print(
    "Churn rate in top 100:",
    round(top_customers["actual_churned"].mean() * 100, 2),
    "%"
)


# ==========================================
# 11. RETRAIN BEST MODEL ON TRAINING + VALIDATION
# ==========================================

# Select the model with the highest validation PR-AUC.
best_model_name = results_df.iloc[0]["model"]

if best_model_name == "SVM linear (no scaling)":
    final_model = svm_unscaled
elif best_model_name == "SVM linear (scaled)":
    final_model = svm_scaled
else:
    final_model = logistic_pipeline

print("\nSelected model:", best_model_name)

# Combine training and validation data.
X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])

# Retrain the selected model.
final_model.fit(X_final, y_final)

print("Final model retrained on training + validation data.")


# ==========================================
# 12. FEATURE WEIGHTS / MODEL COEFFICIENTS
# ==========================================

model_object = final_model.named_steps["model"]

if hasattr(model_object, "coef_"):

    feature_names = (
        final_model.named_steps["prep"]
        .get_feature_names_out()
    )

    weights = pd.Series(
        model_object.coef_[0],
        index=feature_names
    ).sort_values(ascending=False)

    print("\n===== TOP CHURN FACTORS =====")
    print(weights.head(10))

    print("\n===== TOP RETENTION FACTORS =====")
    print(weights.tail(10))

    weights.to_csv("reports/svm_feature_weights.csv")

else:
    print("Selected model does not expose linear coefficients.")


# ==========================================
# 13. SAVE FINAL TRAINED MODEL
# ==========================================

joblib.dump(
    final_model,
    "models/linear_svm_pipeline.joblib"
)

print("\nModel saved successfully!")
print("Selected model:", best_model_name)
print("File: models/linear_svm_pipeline.joblib")
print("File: models/linear_svm_pipeline.joblib")

print("\nGenerated reports:")
print("- reports/model_comparison.csv")
print("- reports/test_top_100_call_list.csv")
print("- reports/svm_feature_weights.csv")

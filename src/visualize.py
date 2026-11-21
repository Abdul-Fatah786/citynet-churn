"""
CityNet Churn Project - Static Visualizations

Run from the project root with:
    python src/visualize.py

Or use your virtual environment's Python executable.

Creates PNG charts in reports/figures/.
"""

from pathlib import Path
import re
import traceback

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

DATA_FILE = DATA_DIR / "citynet_churn.csv"
MODEL_FILE = PROJECT_DIR / "models" / "linear_svm_pipeline.joblib"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COLUMN = "churned"
ID_COLUMN = "customer_id"

DROP_FEATURE_COLUMNS = [
    ID_COLUMN,
    "cancel_request_logged",
    TARGET_COLUMN,
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """Find a column while ignoring case, spaces, and punctuation."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in df.columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]

    return None


def save_chart(filename):
    """
    Save the current figure using a binary file handle.
    This avoids Matplotlib/Pillow path handling issues on some Windows setups.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (FIGURES_DIR / filename).resolve()

    fig = plt.gcf()
    fig.tight_layout()

    try:
        # Saving through an opened binary file handle is more robust on Windows.
        with output_path.open("wb") as output_file:
            fig.savefig(
                output_file,
                format="png",
                dpi=300,
                bbox_inches="tight",
            )

        print(f"[SAVED] {output_path}")

    except OSError as exc:
        # Try a simple alternate filename in the same output directory.
        alternate_path = output_path.with_name(
            f"{output_path.stem}_chart.png"
        )

        print(f"[WARNING] Could not save {output_path.name}: {exc}")
        print(f"[INFO] Trying alternate filename: {alternate_path.name}")

        with alternate_path.open("wb") as output_file:
            fig.savefig(
                output_file,
                format="png",
                dpi=300,
                bbox_inches="tight",
            )

        print(f"[SAVED] {alternate_path}")

    finally:
        plt.close(fig)


def load_csv_if_available(path, description):
    """Load a report CSV, or return None with a useful message."""
    if not path.exists():
        print(f"[SKIPPED] {description}: file not found: {path}")
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[SKIPPED] Could not read {path.name}: {exc}")
        return None


def run_chart(chart_name, chart_function):
    """Run one chart safely so one failure does not stop the others."""
    try:
        chart_function()
    except Exception as exc:
        plt.close("all")
        print(f"\n[ERROR] {chart_name} could not be generated.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print("The remaining visualizations will continue.")


# ============================================================
# 3. CUSTOMER CHURN DISTRIBUTION
# ============================================================

def plot_churn_distribution():
    print("\n1/6 - Customer churn distribution")

    data = load_csv_if_available(DATA_FILE, "Historical dataset")
    if data is None:
        return

    if TARGET_COLUMN not in data.columns:
        print(f"[SKIPPED] '{TARGET_COLUMN}' column not found.")
        return

    counts = data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()

    labels = ["No Churn (0)", "Churn (1)"]
    values = counts.values

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)

    ax.set_title("CityNet Customer Churn Distribution")
    ax.set_xlabel("Customer class")
    ax.set_ylabel("Number of customers")
    ax.set_ylim(0, max(values) * 1.15 if max(values) else 1)

    for bar, value in zip(bars, values):
        percentage = (value / total * 100) if total else 0
        ax.annotate(
            f"{value:,}\n({percentage:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    save_chart("churn_distribution.png")


# ============================================================
# 4. MODEL VALIDATION PR-AUC COMPARISON
# ============================================================

def plot_model_comparison():
    print("\n2/6 - Model comparison")

    path = REPORT_DIR / "model_comparison.csv"
    results = load_csv_if_available(path, "Model comparison")
    if results is None:
        return

    model_col = find_column(results, ["model", "model_name", "classifier"])
    score_col = find_column(
        results,
        ["validation_pr_auc", "val_pr_auc", "pr_auc", "pr-auc"],
    )

    if model_col is None or score_col is None:
        print("[SKIPPED] Could not identify model and validation PR-AUC columns.")
        print("Columns found:", results.columns.tolist())
        return

    results[score_col] = pd.to_numeric(results[score_col], errors="coerce")
    results = results.dropna(subset=[score_col])

    if results.empty:
        print("[SKIPPED] No numeric PR-AUC values found.")
        return

    results = results.sort_values(score_col, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(results[model_col].astype(str), results[score_col])

    ax.set_title("Model Comparison - Validation PR-AUC")
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_ylim(0, max(1.0, results[score_col].max() * 1.15))
    ax.tick_params(axis="x", labelrotation=15)

    for bar, value in zip(bars, results[score_col]):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    save_chart("model_comparison.png")


# ============================================================
# 5. SVM COEFFICIENTS
# ============================================================

def plot_feature_weights():
    print("\n3/6 - SVM feature weights")

    path = REPORT_DIR / "svm_feature_weights.csv"
    weights = load_csv_if_available(path, "SVM feature weights")
    if weights is None:
        return

    feature_col = find_column(weights, ["feature", "feature_name", "name"])
    coefficient_col = find_column(
        weights, ["coefficient", "coef", "weight", "svm_weight"]
    )

    if feature_col is None or coefficient_col is None:
        print("[SKIPPED] Could not identify feature and coefficient columns.")
        print("Columns found:", weights.columns.tolist())
        return

    weights[coefficient_col] = pd.to_numeric(
        weights[coefficient_col], errors="coerce"
    )
    weights = weights.dropna(subset=[coefficient_col]).copy()

    if weights.empty:
        print("[SKIPPED] No numeric coefficients found.")
        return

    weights["absolute_weight"] = weights[coefficient_col].abs()
    top = weights.nlargest(15, "absolute_weight").sort_values(coefficient_col)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top[feature_col].astype(str), top[coefficient_col])
    ax.axvline(0, linewidth=1)

    ax.set_title("Top 15 SVM Coefficients")
    ax.set_xlabel("Coefficient (raw model scale)")
    ax.set_ylabel("Transformed feature")

    save_chart("feature_weights.png")


# ============================================================
# 6. NEW CUSTOMER SVM DECISION SCORES
# ============================================================

def plot_prediction_risk_scores():
    print("\n4/6 - New customer risk scores")

    path = REPORT_DIR / "predictions.csv"
    predictions = load_csv_if_available(path, "Predictions")
    if predictions is None:
        return

    customer_col = find_column(predictions, ["customer_id"])
    score_col = find_column(
        predictions, ["svm_score", "decision_score", "risk_score"]
    )

    if customer_col is None or score_col is None:
        print("[SKIPPED] Could not identify customer_id and svm_score columns.")
        print("Columns found:", predictions.columns.tolist())
        return

    predictions[score_col] = pd.to_numeric(
        predictions[score_col], errors="coerce"
    )
    predictions = predictions.dropna(subset=[score_col]).sort_values(score_col)

    if predictions.empty:
        print("[SKIPPED] No numeric decision scores found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(predictions[customer_col].astype(str), predictions[score_col])
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label="SVM decision boundary (0)",
    )

    ax.set_title("New Customers - SVM Decision Scores")
    ax.set_xlabel("SVM decision score (not a probability)")
    ax.set_ylabel("Customer ID")
    ax.legend()

    save_chart("prediction_risk_scores.png")


# ============================================================
# 7. ACTUAL OUTCOMES IN TOP-100 TEST CUSTOMER LIST
# ============================================================

def plot_top_100_outcomes():
    print("\n5/6 - Top 100 test customer outcomes")

    path = REPORT_DIR / "test_top_100_call_list.csv"
    top100 = load_csv_if_available(path, "Top-100 test call list")
    if top100 is None:
        return

    actual_col = find_column(
        top100, ["churned", "actual_churn", "actual", "y_true"]
    )

    if actual_col is None:
        print("[SKIPPED] Could not identify the actual target column.")
        print("Columns found:", top100.columns.tolist())
        return

    actual = pd.to_numeric(top100[actual_col], errors="coerce").dropna()
    counts = actual.value_counts().reindex([0, 1], fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["No Churn (0)", "Churn (1)"], counts.values)

    ax.set_title("Actual Outcomes - Top 100 Test Customers")
    ax.set_xlabel("Actual customer class")
    ax.set_ylabel("Number of customers")
    ax.set_ylim(0, max(counts.values) * 1.15 if max(counts.values) else 1)

    for bar, value in zip(bars, counts.values):
        ax.annotate(
            f"{value}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    save_chart("top_100_customer_outcomes.png")


# ============================================================
# 8. CONFUSION MATRIX - SAVED FINAL MODEL ON HELD-OUT TEST SET
# ============================================================

def plot_confusion_matrix():
    print("\n6/6 - Confusion matrix")

    if not MODEL_FILE.exists():
        print(f"[SKIPPED] Saved model not found: {MODEL_FILE}")
        return

    data = load_csv_if_available(DATA_FILE, "Historical dataset")
    if data is None:
        return

    required = {TARGET_COLUMN, ID_COLUMN, "cancel_request_logged"}
    missing = required.difference(data.columns)
    if missing:
        print(f"[SKIPPED] Required dataset columns missing: {sorted(missing)}")
        return

    # Recreate the original stratified 60/20/20 split.
    X = data.drop(columns=DROP_FEATURE_COLUMNS)
    y = data[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # The saved pipeline was fitted on train + validation.
    # Evaluate it on the untouched test partition.
    final_model = joblib.load(MODEL_FILE)
    y_pred = final_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn (0)", "Churn (1)"],
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("CityNet - Final SVM Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    save_chart("confusion_matrix.png")

    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix:")
    print(cm)
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")

    accuracy = (tn + tp) / cm.sum() if cm.sum() else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


# ============================================================
# 9. RUN ALL VISUALIZATIONS
# ============================================================

def main():
    print("=" * 55)
    print("CITYNET CHURN - STATIC VISUALIZATION GENERATOR")
    print("=" * 55)

    charts = [
        ("Customer churn distribution", plot_churn_distribution),
        ("Model comparison", plot_model_comparison),
        ("SVM feature weights", plot_feature_weights),
        ("New customer risk scores", plot_prediction_risk_scores),
        ("Top 100 test customer outcomes", plot_top_100_outcomes),
        ("Confusion matrix", plot_confusion_matrix),
    ]

    for chart_name, chart_function in charts:
        run_chart(chart_name, chart_function)

    print("\n" + "=" * 55)
    print("VISUALIZATION GENERATION FINISHED")
    print(f"Output folder: {FIGURES_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()

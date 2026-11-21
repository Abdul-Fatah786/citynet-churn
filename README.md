# CityNet Customer Churn Prediction

## Project Overview

CityNet Customer Churn Prediction is a machine learning project that identifies customers who are more likely to cancel their subscriptions.

The system helps the retention team prioritize customers for follow-up calls.

## User Story

As a retention manager, I want a weekly list of the 100 customers most likely to cancel next month, so my team can contact them first.

## Dataset

- Historical records: 5,000 customers
- Features: 11
- Target: `churned`
- Positive class: 1 (customer churned)
- Negative class: 0 (customer did not churn)

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Machine Learning Workflow

1. Data loading and exploration
2. Missing-value handling
3. Numeric and categorical feature identification
4. One-hot encoding of categorical features
5. Model training and comparison
6. Validation-based model selection
7. Final model training using training and validation data
8. Customer churn prediction and risk ranking

## Models Compared

- Logistic Regression
- Linear SVM with scaling
- Linear SVM without scaling

The model with the highest validation PR-AUC is selected.

## Data Split

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 60% | 3,000 |
| Validation | 20% | 1,000 |
| Testing | 20% | 1,000 |

## Validation Results

| Model | PR-AUC |
|---|---:|
| Linear SVM without scaling | 0.3886 |
| Logistic Regression | 0.3826 |
| Linear SVM with scaling | 0.3824 |

## Test Results

The existing test evaluation was performed on the scaled Linear SVM:

- Test PR-AUC: 0.3918
- Actual churners in top 100: 52
- Expected churners in random 100: approximately 17.9

These test results belong to the scaled SVM, not the validation-selected unscaled SVM.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and save the selected model:

```bash
python src/lab7_save.py
```

Generate predictions:

```bash
python src/predict.py
```

Or provide another CSV file:

```bash
python src/predict.py data/new_customers.csv
```

Predictions are saved to:

`reports/predictions.csv`

## Intended Use

- Identify customers who may be at risk of cancellation.
- Prioritize customer retention calls.
- Support retention teams in making data-informed decisions.

## Not For

- Automatically cancelling customer accounts.
- Making decisions without human review.
- Treating model scores as guaranteed churn probabilities.

## Assumptions

- Historical customer behavior can provide useful signals about future churn.
- The `churned` column represents cancellation in the following month.
- New customer records contain the same required features as the training data.

## Risks and Limitations

- Model predictions are not guaranteed to be correct.
- SVM decision scores are not probabilities.
- Customer behavior may change over time.
- Missing or inconsistent input data can affect predictions.
- Model performance should be monitored as new customer data becomes available.
- Feature relationships do not establish causation.
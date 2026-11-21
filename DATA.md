# Data Card: CityNet Churn Dataset

## Dataset Information

- Source: CityNet customer records (synthetic data created for this course).
- Rows: 5,000
- Columns: 14
- One row represents one customer measured over the previous 90 days.

## Target Variable

Target column: `churned`

- `1`: Customer cancelled in the next month.
- `0`: Customer did not cancel in the next month.

Target distribution:

| Class | Customers | Percentage |
|---|---:|---:|
| No Churn (0) | 4,104 | 82.08% |
| Churn (1) | 896 | 17.92% |

## Missing Values

- `payment_method`: 149 missing values.
- `age`: 411 missing values.

Handling strategy:

- Numeric features: missing values filled using median imputation.
- Categorical features: missing values filled using most-frequent imputation.

## Features Used

The model uses 11 input features:

- region
- plan
- monthly_bill
- tv_bundle
- tenure_months
- support_calls_90d
- outage_hours_90d
- payment_method
- auto_pay
- paid_late_last_bill
- age

## Columns Not Used

- `customer_id`: Identifier only; not a predictive feature.
- `cancel_request_logged`: Excluded because it may reveal information closely related to the target, creating target leakage.
- `churned`: Target variable, not an input feature.

## Preprocessing

- Numeric columns are imputed using median values.
- Categorical columns are imputed using their most frequent values.
- Categorical features are transformed using One-Hot Encoding.
- Numeric scaling is applied only in the scaled model pipeline.

## Known Issues and Limitations

- The dataset is synthetic and does not represent verified real-world CityNet customer records.
- The target classes are imbalanced.
- Missing values exist in some columns.
- Model performance on synthetic data may not generalize to real customer behavior.
- Customer associations identified by the model should not be interpreted as causal relationships.
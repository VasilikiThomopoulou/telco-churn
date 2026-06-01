# Telco Customer Churn Prediction

Predicting customer churn with a scikit-learn pipeline (scaling + one-hot
encoding + gradient boosting), served as a Streamlit dashboard.

## What it does
- **Insights tab** — feature importances showing what drives churn
- **Predict tab** — enter a customer's details, get their churn probability

Model: gradient boosting, ROC-AUC ≈ 0.84 on held-out data.

---
title: Telco Churn Predictor
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# Telco Customer Churn Prediction

Predicting customer churn with a scikit-learn pipeline (scaling + one-hot
encoding + gradient boosting), served as a Streamlit dashboard.

## What it does
- **Insights tab** — feature importances showing what drives churn
- **Predict tab** — enter a customer's details, get their churn probability

Model: gradient boosting, ROC-AUC ≈ 0.84 on held-out data.
"""
Streamlit churn dashboard
Loads the fitted pipeline and serves and Insights tab and a predict tab
Run from project root with: streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/telco_churn_model.pkl")

pipe = load_model()

st.title("Telco Customer Churn Dashboard")

tab_insights, tab_predict = st.tabs(["Insights", "Predict"])

#---------------Insights Tab----------------
with tab_insights:
    st.header("Feature Importance")

    pre = pipe.named_steps['pre']
    model = pipe.named_steps['model']
    names = pre.get_feature_names_out()

    importances = (
        pd.Series(model.feature_importances_, index=names)
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(importances)

    st.markdown(
        """
        The model finds that **contract type**
        (month-to-month), **tenure / total charges** (how long they've been a
        customer), **monthly charges**, and **fiber-optic internet** are the
        strongest churn signals.

        **Recommendation:** target month-to-month, short-tenure, high-bill
        customers with incentives to move onto annual contracts. Contract type
        is the single biggest lever.
        """
    )

#---------------Predict Tab----------------
with tab_predict:
    st.header("Predict Customer Churn")
    st.caption("Enter customer details to predict the probability of churn")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
        total = st.number_input("Total charges ($)", 0.0, 9000.0, 800.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

    with col2:
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])

    with col3:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiline = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        online_sec = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_bak = st.selectbox("Online backup", ["Yes", "No", "No internet service"])

    # Remaining columns the model expects — set sensible defaults so we send all 19
    device_prot = "No"
    tech_support = "No"
    streaming_tv = "No"
    streaming_movies = "No"

    if st.button("Predict", type="primary"):
        # Build a ONE-ROW DataFrame with the exact 19 training columns
        row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiline,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        prob = pipe.predict_proba(row)[0, 1]   # probability of churn for this one customer

        st.metric("Churn probability", f"{prob:.1%}")
        if prob >= 0.5:
            st.error("⚠️ High churn risk — consider a retention offer.")
        else:
            st.success("✅ Low churn risk.")

  
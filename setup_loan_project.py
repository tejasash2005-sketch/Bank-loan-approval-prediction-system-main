import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Setup folders
# -----------------------------
folders = [
    "Loan-Approval-System/data",
    "Loan-Approval-System/models",
    "Loan-Approval-System/logs",
    "Loan-Approval-System/src",
    "Loan-Approval-System/app"
]
for f in folders:
    os.makedirs(f, exist_ok=True)

# -----------------------------
# Generate dataset (24 features all contribute)
# -----------------------------
np.random.seed(42)
n_samples = 500
X = np.random.rand(n_samples, 24) * 10000
weights = np.arange(1, 25)  # weight each feature differently
y_score = np.dot(X, weights)
y = (y_score > np.median(y_score)).astype(int)
columns = [f"f{i+1}" for i in range(24)]
df = pd.DataFrame(X, columns=columns)
df['target'] = y
df.to_csv("Loan-Approval-System/data/loan_data.csv", index=False)

# -----------------------------
# Train models
# -----------------------------
log_path = "Loan-Approval-System/models/logistic_model.pkl"
ensemble_path = "Loan-Approval-System/models/ensemble_model.pkl"

log_model = LogisticRegression(max_iter=500)
log_model.fit(X, y)
pickle.dump(log_model, open(log_path, "wb"))

ensemble_model = RandomForestClassifier(n_estimators=100, random_state=42)
ensemble_model.fit(X, y)
pickle.dump(ensemble_model, open(ensemble_path, "wb"))

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Advanced Bank Loan Approval System", layout="wide")
st.title("Advanced Bank Loan Approval System")

# Load models
log_model = pickle.load(open(log_path, "rb"))
ensemble_model = pickle.load(open(ensemble_path, "rb"))

# Sidebar: 24 features
st.sidebar.header("Applicant Features")
feature_labels = [
    "Applicant Income", "Coapplicant Income", "Loan Amount", "Credit History",
    "Total Income", "Loan-to-Income Ratio", "Log Applicant Income", "Log Coapplicant Income",
    "Log Total Income", "Loan per Coapplicant", "DTI Ratio", "Credit-Income Interaction",
    "Applicant Income Squared", "Loan Amount Squared", "Income Ratio", "Loan-Credit Interaction",
    "High Loan Flag", "High Income Flag", "Coapplicant Flag", "Loan Income Log Ratio",
    "Sqrt Applicant Income", "Sqrt Coapplicant Income", "Applicant-Loan Interaction", "Coapplicant-Loan Interaction"
]

features_dict = {}
for label in feature_labels:
    if "Credit History" in label or "Flag" in label:
        features_dict[label] = st.sidebar.selectbox(label, [0, 1], index=0)
    else:
        features_dict[label] = st.sidebar.number_input(label, value=1000.0, step=1.0)

X_input_df = pd.DataFrame([list(features_dict.values())], columns=feature_labels)
X_input_array = X_input_df.to_numpy()

# -----------------------------
# Personal Information
# -----------------------------
st.subheader("Personal Information")
cols = st.columns(5)
personal_info = {}
personal_info["Name"] = cols[0].text_input("Full Name", "")
personal_info["Age"] = cols[1].number_input("Age", min_value=18, max_value=100, value=30)
personal_info["Gender"] = cols[2].selectbox("Gender", ["Male", "Female", "Other"])
personal_info["Nationality"] = cols[3].text_input("Nationality", "Unknown")
personal_info["Marital Status"] = cols[4].selectbox("Marital Status", ["Single", "Married", "Other"])
st.write(personal_info)

# -----------------------------
# Applicant Feature Table
# -----------------------------
st.subheader("Applicant Features")
st.dataframe(X_input_df)

# -----------------------------
# Predict Loan Approval
# -----------------------------
if st.button("Predict Loan Approval"):
    prob = ensemble_model.predict_proba(X_input_array)[0][1]
    prediction = 1 if prob > 0.5 else 0
    risk = "Low" if prob > 0.8 else ("Medium" if prob > 0.5 else "High")

    st.subheader("Loan Prediction")
    st.success("Loan Approved ✅" if prediction==1 else "Loan Rejected ❌")
    st.write(f"Probability: {prob:.2f}")
    st.write(f"Risk: {risk}")

    # Fraud detection demo
    iso = IsolationForest(contamination=0.05, random_state=42)
    X_demo = np.array([[0,0,0,1],[1000,2000,500,0],[2000,0,100,1],[1500,500,200,1]])
    iso.fit(X_demo)
    fraud_flag = iso.predict([X_input_array[0][:4]])[0]==-1
    st.subheader("Fraud Check")
    st.write("⚠ Fraud Alert!" if fraud_flag else "No fraud detected")

    # -----------------------------
    # SHAP explanation
    # -----------------------------
    explainer = shap.TreeExplainer(ensemble_model)
    shap_values = explainer.shap_values(X_input_df)

    # Handle multi-class/tree output
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1][0]
        base_val = float(explainer.expected_value[1])
    else:
        shap_vals_class1 = shap_values[0][0]
        base_val = float(explainer.expected_value)

    order = np.argsort(np.abs(shap_vals_class1))[::-1]
    shap_exp_sorted = shap.Explanation(
        values=shap_vals_class1[order],
        base_values=base_val,
        data=X_input_df.iloc[0, order],
        feature_names=X_input_df.columns[order]
    )

    # Waterfall plot (top 10 features)
    st.subheader("SHAP Waterfall (Top Features)")
    fig, ax = plt.subplots(figsize=(14,6))
    shap.plots.waterfall(shap_exp_sorted, show=False, max_display=10)
    st.pyplot(fig)

    # -----------------------------
    # Bottom section: Full SHAP bar plot for all 24 features
    # -----------------------------
    st.subheader("SHAP Feature Impact (All 24 Features)")
    fig2, ax2 = plt.subplots(figsize=(14,6))
    shap.plots.bar(shap_exp_sorted, max_display=24)
    st.pyplot(fig2)
# streamlit_loan_app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Advanced Bank Loan Approval System", layout="wide")
st.title("📊 Advanced Bank Loan Approval System")

# Ensure data folder exists
os.makedirs("data", exist_ok=True)
data_file = "data/applicants.csv"

# -----------------------------
# Load Models
# -----------------------------
try:
    log_model = pickle.load(open("models/logistic_model.pkl", "rb"))
    ensemble_model = pickle.load(open("models/ensemble_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'models/logistic_model.pkl' and 'models/ensemble_model.pkl' exist.")
    st.stop()

# -----------------------------
# Sidebar: Applicant Features
# -----------------------------
st.sidebar.header("Applicant Features")
feature_labels = [
    'Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Credit History', 'Total Income',
    'Loan-to-Income Ratio', 'Log Applicant Income', 'Log Coapplicant Income', 'Log Total Income',
    'Loan per Coapplicant', 'DTI Ratio', 'Credit-Income Interaction', 'Applicant Income Squared',
    'Loan Amount Squared', 'Income Ratio', 'Loan-Credit Interaction', 'High Loan Flag', 'High Income Flag',
    'Coapplicant Flag', 'Loan Income Log Ratio', 'Sqrt Applicant Income', 'Sqrt Coapplicant Income',
    'Applicant-Loan Interaction', 'Coapplicant-Loan Interaction', 'Marital Status Flag',
    'Gender Flag', 'Age', 'Nationality Flag', 'Employment Status Flag'
]

features_dict = {}
for label in feature_labels:
    if "Flag" in label or "Credit History" in label:
        features_dict[label] = st.sidebar.selectbox(label, [0, 1], index=0)
    elif "Marital Status" in label:
        features_dict[label] = st.sidebar.selectbox(label, ["Single", "Married", "Other"], index=0)
    elif "Gender" in label:
        features_dict[label] = st.sidebar.selectbox(label, ["Male", "Female", "Other"], index=0)
    elif "Nationality" in label:
        features_dict[label] = st.sidebar.text_input(label, "Unknown")
    else:
        features_dict[label] = st.sidebar.number_input(label, value=1000.0, step=1.0)

X_input_df = pd.DataFrame([list(features_dict.values())], columns=feature_labels)
X_input_array = X_input_df.to_numpy()

# -----------------------------
# Personal Info
# -----------------------------
st.subheader("👤 Personal Information")
cols = st.columns(5)
personal_info = {}
personal_info["Name"] = cols[0].text_input("Full Name", "")
personal_info["Age"] = cols[1].number_input("Age", min_value=18, max_value=100, value=30)
personal_info["Gender"] = cols[2].selectbox("Gender", ["Male","Female","Other"])
personal_info["Nationality"] = cols[3].text_input("Nationality", "Unknown")
personal_info["Marital Status"] = cols[4].selectbox("Marital Status", ["Single","Married","Other"])
st.write(personal_info)

# -----------------------------
# Applicant Feature Table
# -----------------------------
st.subheader("📋 Applicant Features Overview")
st.dataframe(X_input_df)

# -----------------------------
# Predict Loan Approval
# -----------------------------
if st.button("Predict Loan Approval"):
    # Model Prediction
    prob = ensemble_model.predict_proba(X_input_array)[0][1]
    prediction = 1 if prob>0.5 else 0
    risk = "Low" if prob>0.8 else ("Medium" if prob>0.5 else "High")

    st.subheader("💡 Loan Prediction")
    st.success("Loan Approved ✅" if prediction==1 else "Loan Rejected ❌")
    st.write(f"Probability of Approval: {prob:.2f}")
    st.write(f"Risk Level: {risk}")

    # Fraud Detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(np.array([[0,0,0,1],[1000,2000,500,0],[2000,0,100,1],[1500,500,200,1]]))
    fraud_flag = iso.predict([X_input_array[0][:4]])[0]==-1
    st.subheader("⚠ Fraud Check")
    st.write("⚠ Fraud Alert!" if fraud_flag else "No fraud detected")

    # -----------------------------
    # Save Applicant Data
    # -----------------------------
    record = {**personal_info, **features_dict}
    record["Prediction"] = prediction
    record["Probability"] = prob
    record["Risk Level"] = risk
    record["Fraud Alert"] = int(fraud_flag)

    if os.path.exists(data_file):
        df_existing = pd.read_csv(data_file)
        df_new = pd.DataFrame([record])
        df_save = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_save = pd.DataFrame([record])

    df_save.to_csv(data_file, index=False)
    st.success("✅ Applicant data saved successfully!")

    # -----------------------------
    # Interactive Plots
    # -----------------------------
    st.subheader("📈 Feature Dashboard")

    # 1️⃣ Binary flags pie chart
    flag_cols = [col for col in feature_labels if "Flag" in col or "Credit History" in col]
    flag_values = X_input_df[flag_cols].iloc[0].value_counts()
    fig1 = go.Figure(data=[go.Pie(labels=flag_values.index, values=flag_values.values, hole=0.4)])
    fig1.update_layout(title_text="Binary Feature Distribution", width=400, height=400)
    st.plotly_chart(fig1, use_container_width=False)

    # 2️⃣ Numeric bar chart
    numeric_cols = [col for col in feature_labels if col not in flag_cols]
    fig2 = px.bar(x=numeric_cols, y=X_input_df[numeric_cols].iloc[0].values,
                  labels={"x":"Feature","y":"Value"},
                  text=X_input_df[numeric_cols].iloc[0].values,
                  title="All Numeric Feature Values")
    fig2.update_layout(xaxis_tickangle=-45, width=1200, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Prediction probability pie
    fig3 = go.Figure(data=[go.Pie(labels=["Approved","Rejected"], values=[prob,1-prob],
                                  hole=0.3, marker_colors=["#4CAF50","#F44336"])])
    fig3.update_layout(title_text="Prediction Probability Distribution", width=400, height=400)
    st.plotly_chart(fig3, use_container_width=False)

    # 4️⃣ Feature value distribution (histogram of all numeric features)
    fig4 = px.bar(X_input_df, y=numeric_cols,
                  labels={"value":"Value","variable":"Feature"},
                  title="Feature Value Distribution (All Numeric Features)")
    st.plotly_chart(fig4, use_container_width=True)

    # 5️⃣ Fraud alert pie
    fig5 = go.Figure(data=[go.Pie(labels=["Fraud","No Fraud"], values=[int(fraud_flag), int(not fraud_flag)],
                                  hole=0.3, marker_colors=["#FF5722","#2196F3"])])
    fig5.update_layout(title_text="Fraud Alert Status", width=400, height=400)
    st.plotly_chart(fig5, use_container_width=False)
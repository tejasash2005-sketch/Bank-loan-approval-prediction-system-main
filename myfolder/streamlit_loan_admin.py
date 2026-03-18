# streamlit_loan_admin.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Admin Dashboard", layout="wide")
st.title("🗂 Advanced Loan Admin Dashboard")

data_file = "data/applicants.csv"

if os.path.exists(data_file):
    df = pd.read_csv(data_file)

    # -----------------------------
    # Reorder columns: Personal Info first
    personal_cols = ["Name", "Age", "Gender", "Nationality", "Marital Status"]
    other_cols = [col for col in df.columns if col not in personal_cols]
    df = df[personal_cols + other_cols]

    st.subheader("All Applicant Records")
    st.dataframe(df)

    # -----------------------------
    # Filters
    st.subheader("Filter Applicants")
    filter_pred = st.selectbox("Prediction", ["All", "Approved", "Rejected"])
    filter_risk = st.selectbox("Risk Level", ["All"] + df["Risk Level"].unique().tolist())
    search_name = st.text_input("Search by Name")

    df_filtered = df.copy()
    if filter_pred != "All":
        df_filtered = df_filtered[df_filtered["Prediction"] == (1 if filter_pred=="Approved" else 0)]
    if filter_risk != "All":
        df_filtered = df_filtered[df_filtered["Risk Level"] == filter_risk]
    if search_name:
        df_filtered = df_filtered[df_filtered["Name"].str.contains(search_name, case=False)]

    st.subheader("Filtered Applicant Records")
    st.dataframe(df_filtered)

    # -----------------------------
    # Interactive Charts
    st.subheader("📊 Applicant Data Overview")

    col1, col2 = st.columns(2)

    # 1️⃣ Approval vs Rejection Pie
    with col1:
        pred_counts = df["Prediction"].value_counts().reindex([1,0], fill_value=0)
        fig1 = go.Figure(data=[go.Pie(
            labels=["Approved","Rejected"],
            values=pred_counts.values,
            hole=0.4,
            marker_colors=["#4CAF50","#F44336"]
        )])
        fig1.update_layout(title_text="Approval vs Rejection", width=400, height=400)
        st.plotly_chart(fig1, use_container_width=False)

    # 2️⃣ Risk Level Distribution Bar
    with col2:
        risk_counts = df["Risk Level"].value_counts()
        fig2 = px.bar(x=risk_counts.index, y=risk_counts.values,
                      labels={"x":"Risk Level","y":"Number of Applicants"},
                      title="Risk Level Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Average Numeric Feature Values
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64','float64'] and col not in ["Prediction"]]
    if numeric_cols:
        avg_vals = df[numeric_cols].mean()
        fig3 = px.bar(x=avg_vals.index, y=avg_vals.values,
                      labels={"x":"Feature","y":"Average Value"},
                      title="Average Numeric Feature Values (All Applicants)")
        fig3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

    # 4️⃣ Binary Flags and Credit History Pie
    flag_cols = [col for col in df.columns if "Flag" in col or "Credit History" in col]
    if flag_cols:
        flag_sums = df[flag_cols].sum()
        fig4 = go.Figure(data=[go.Pie(labels=flag_sums.index, values=flag_sums.values,
                                      hole=0.3)])
        fig4.update_layout(title_text="Binary Flags / Credit History Summary", width=400, height=400)
        st.plotly_chart(fig4, use_container_width=False)

else:
    st.info("No applicant data found. Please submit applications from the main loan app first.")
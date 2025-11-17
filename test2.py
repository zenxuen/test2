import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.express as px

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

st.title("💼 Salary Prediction Dashboard (2021–2025)")


# ---------------------------------------------------------
# Load Internal Dataset (NO UPLOAD)
# ---------------------------------------------------------
data = {
    "work_year": [2022, 2022, 2022, 2022, 2022, 2022, 2022],
    "experience_level": ["EN", "MI", "MI", "MI", "EN", "EX", "SE"],
    "employment_type": ["FT", "FT", "FT", "FT", "CT", "FT", "FT"],
    "job_title": [
        "CYBER PROGRAM MANAGER",
        "SECURITY ANALYST",
        "SECURITY ANALYST",
        "IT SECURITY ANALYST",
        "CYBER SECURITY ANALYST",
        "APPLICATION SECURITY ARCHITECT",
        "SECURITY RESEARCHER"
    ],
    "salary_in_usd": [63000, 95000, 70000, 48853, 120000, 315000, 220000],
    "company_size": ["S", "M", "M", "L", "S", "L", "M"]
}

df = pd.DataFrame(data)


# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------
X = df[["work_year", "job_title", "experience_level", "company_size"]]
y = df["salary_in_usd"]

categorical_cols = ["job_title", "experience_level", "company_size"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

model.fit(X, y)


# ---------------------------------------------------------
# Forecast 2021–2025
# ---------------------------------------------------------
future_years = np.arange(2021, 2026)
sample_job = df["job_title"].mode()[0]
sample_exp = df["experience_level"].mode()[0]
sample_size = df["company_size"].mode()[0]

future_data = pd.DataFrame({
    "work_year": future_years,
    "job_title": sample_job,
    "experience_level": sample_exp,
    "company_size": sample_size
})

future_predictions = model.predict(future_data)

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Predicted Salary (USD)": future_predictions
})

# ---------------------------------------------------------
# Plot Chart
# ---------------------------------------------------------
st.subheader("📈 Predicted Salary Trend (2021–2025)")
fig = px.line(
    forecast_df,
    x="Year",
    y="Predicted Salary (USD)",
    markers=True,
    title="Salary Prediction (2021–2025)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Custom User Prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Salary for Custom Job")

col1, col2, col3, col4 = st.columns(4)

with col1:
    use_year = st.number_input("Work Year", min_value=2020, max_value=2035, value=2023)

with col2:
    use_job = st.selectbox("Job Title", sorted(df["job_title"].unique()))

with col3:
    use_exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))

with col4:
    use_size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

user_input = pd.DataFrame({
    "work_year": [use_year],
    "job_title": [use_job],
    "experience_level": [use_exp],
    "company_size": [use_size]
})

pred_salary = model.predict(user_input)[0]

st.metric("💰 Predicted Salary (USD)", f"${pred_salary:,.2f}")

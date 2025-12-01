import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Cybersecurity Salary Forecast — ML Model (2020–2030)",
    page_icon="🔐",
    layout="wide"
)

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")

    # --- Clean Columns ---
    df.columns = df.columns.str.strip().str.lower()

    # Standardize expected column names:
    rename_map = {
        "employee_residence": "employee_residence",
        "company_location": "company_location",
        "company_size": "company_size",
        "experience_level": "experience_level",
        "employment_type": "employment_type",
        "job_title": "job_title",
        "salary_in_usd": "salary",
        "work_year": "work_year"
    }
    df = df.rename(columns=rename_map)

    return df

df = load_data()

# =========================================
# FILTERABLE UI OPTIONS
# =========================================
job_list = sorted(df["job_title"].dropna().unique())
exp_list = sorted(df["experience_level"].dropna().unique())
emp_type_list = sorted(df["employment_type"].dropna().unique())
size_list = sorted(df["company_size"].dropna().unique())

# Sidebar
st.sidebar.header("Select Profile for Prediction")

job = st.sidebar.selectbox("Job Title", job_list)
exp = st.sidebar.selectbox("Experience Level", exp_list)
emp_type = st.sidebar.selectbox("Employment Type", emp_type_list)
comp_size = st.sidebar.selectbox("Company Size", size_list)
model_choice = st.sidebar.selectbox("ML Model", ["Random Forest"])

# =========================================
# BUILD INPUT PROFILE
# =========================================
profile = {
    "job_title": job,
    "experience_level": exp,
    "employment_type": emp_type,
    "company_size": comp_size,
}

# =========================================
# FILTER ACTUAL DATA FOR THIS PROFILE
# =========================================
mask = (
    (df["job_title"] == job) &
    (df["experience_level"] == exp) &
    (df["employment_type"] == emp_type) &
    (df["company_size"] == comp_size)
)

actual_df = df[mask]

# If no data found → Use whole dataset
if actual_df.empty:
    st.warning("This profile has **no actual data** in 2020–2022. Showing ML predictions only.")
    actual_df = df.copy()

# =========================================
# TRAIN ML MODEL (use entire dataset but learn trends)
# =========================================
features = ["job_title", "experience_level", "employment_type", "company_size", "work_year"]
target = "salary"

X = df[features]
y = df[target]

# Encoding
categorical_cols = ["job_title", "experience_level", "employment_type", "company_size"]
numeric_cols = ["work_year"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = RandomForestRegressor(n_estimators=400, random_state=42)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

pipe.fit(X, y)

# =========================================
# MAKE FULL FORECAST (2020–2030)
# =========================================
years = list(range(2020, 2030 + 1))
forecast_rows = []

for yr in years:
    row = {
        "job_title": job,
        "experience_level": exp,
        "employment_type": emp_type,
        "company_size": comp_size,
        "work_year": yr
    }

    pred = pipe.predict(pd.DataFrame([row]))[0]
    source = "ML Predicted"

    # Use actual data if 2020–2022 exist
    if yr in actual_df["work_year"].values:
        pred = actual_df[actual_df["work_year"] == yr]["salary"].mean()
        source = "Actual"

    forecast_rows.append([yr, pred, source])

forecast_df = pd.DataFrame(forecast_rows, columns=["year", "salary", "source"])

# =========================================
# UI DISPLAY
# =========================================
st.title("🔐 Cybersecurity Salary Forecast — ML Model (2020–2030)")
st.subheader("Salary Forecast Chart")

import plotly.express as px

fig = px.line(
    forecast_df,
    x="year",
    y="salary",
    markers=True,
    color="source",
    title="Salary Forecast (2020–2030)"
)

st.plotly_chart(fig, use_container_width=True)

# Table
st.subheader("Forecast Data")
st.dataframe(forecast_df, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cybersecurity Salary Forecast",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# LOAD DATA (CACHED)
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("salaries_cyber_clean.csv")

df = load_data()

# =========================================================
# TRAIN ML MODEL (CACHED)
# =========================================================
feature_cols = [
    "work_year",
    "job_title",
    "experience_level",
    "employment_type",
    "company_size"
]

target_col = "salary_in_usd"

cat_cols = ["job_title", "experience_level", "employment_type", "company_size"]
num_cols = ["work_year"]

@st.cache_resource
def train_radius_model(df):
    X = df[feature_cols]
    y = df[target_col]

    preprocess_ml = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    model = Pipeline([
        ("prep", preprocess_ml),
        ("reg", XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2,
            objective="reg:squarederror"
        ))
    ])

    model.fit(X, y)
    return model

radius_ml_model = train_radius_model(df)

# =========================================================
# CACHED FILTER FUNCTIONS
# =========================================================
@st.cache_data
def get_profile_history(job, exp, emp_type, size):
    return df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["employment_type"] == emp_type) &
        (df["company_size"] == size)
    ]

@st.cache_data
def get_similar_year(job, exp, year):
    filt = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["work_year"] == year)
    ]
    if filt.empty:
        return None
    return filt["salary_in_usd"].mean()

# =========================================================
# SIDEBAR UI
# =========================================================
prediction_method = st.sidebar.radio(
    "Salary Prediction Method",
    ["Growth-Based", "Radius"]
)

job = st.sidebar.selectbox("Job Title", sorted(df["job_title"].unique()))
exp = st.sidebar.selectbox("Experience Level", sorted(df["experience_level"].unique()))
emp_type = st.sidebar.selectbox("Employment Type", sorted(df["employment_type"].unique()))
size = st.sidebar.selectbox("Company Size", sorted(df["company_size"].unique()))

all_years = list(range(2020, 2030 + 1))

# =========================================================
# MAIN PREDICTION LOGIC
# =========================================================
def get_salary(year, job, exp, emp_type, size, method):

    # ----------------------------- ML ONLY (RADIUS) -----------------------------
    if method == "Radius":
        input_row = pd.DataFrame({
            "work_year": [year],
            "job_title": [job],
            "experience_level": [exp],
            "employment_type": [emp_type],
            "company_size": [size]
        })
        value = radius_ml_model.predict(input_row)[0]
        return max(0, value), "Predicted (Radius ML)"

    # --------------------------- GROWTH-BASED MODEL ----------------------------
    profile_history = get_profile_history(job, exp, emp_type, size)

    # CASE 1 — Exact year exists
    exact_match = profile_history[profile_history["work_year"] == year]
    if not exact_match.empty:
        return exact_match["salary_in_usd"].iloc[0], "Actual"

    # CASE 2 — Use similar profiles for this year
    similar = get_similar_year(job, exp, year)
    if similar is not None:
        return similar, "Estimated (Similar Roles)"

    # CASE 3 — fallback ML
    fallback = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "employment_type": [emp_type],
        "company_size": [size]
    })
    ml_fallback = radius_ml_model.predict(fallback)[0]
    return ml_fallback, "Fallback ML"

# =========================================================
# RUN FORECAST
# =========================================================
forecast_rows = []

for yr in all_years:
    value, label = get_salary(yr, job, exp, emp_type, size, prediction_method)
    forecast_rows.append({
        "Year": yr,
        "Salary (USD)": round(value, 2),
        "Source": label
    })

df_out = pd.DataFrame(forecast_rows)

# =========================================================
# DISPLAY RESULTS
# =========================================================
st.title("Cybersecurity Salary Forecast")
st.write(f"### Selected Role: {job} — {exp}")

st.line_chart(df_out, x="Year", y="Salary (USD)", height=420)

with st.expander("Raw Forecast Data"):
    st.dataframe(df_out)

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("salaries_cyber_clean.csv")

# ------------------------------------------------------------
# SIDEBAR SELECTION
# ------------------------------------------------------------
st.sidebar.header("Cybersecurity Salary Prediction")

prediction_method = st.sidebar.radio(
    "Salary Prediction Method",
    ["Growth-Based", "Radius"]
)

job = st.sidebar.selectbox("Job Title", sorted(df["job_title"].unique()))
exp = st.sidebar.selectbox("Experience Level", sorted(df["experience_level"].unique()))
emp_type = st.sidebar.selectbox("Employment Type", sorted(df["employment_type"].unique()))
size = st.sidebar.selectbox("Company Size", sorted(df["company_size"].unique()))

start_year = 2020
end_year = 2030
all_years = list(range(start_year, end_year + 1))

# ------------------------------------------------------------
# TRAIN RADIUS ML MODEL (Pure ML, no growth rules)
# ------------------------------------------------------------
feature_cols = [
    "work_year",
    "job_title",
    "experience_level",
    "employment_type",
    "company_size"
]

target_col = "salary_in_usd"

X = df[feature_cols]
y = df[target_col]

cat_cols = ["job_title", "experience_level", "employment_type", "company_size"]
num_cols = ["work_year"]

preprocess_ml = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

radius_ml_model = Pipeline([
    ("prep", preprocess_ml),
    ("reg", XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2,
        objective="reg:squarederror"
    ))
])

radius_ml_model.fit(X, y)

# ------------------------------------------------------------
# GROWTH-BASED SUPPORT FUNCTIONS
# ------------------------------------------------------------
def similar_growth(df, job, exp, year):
    filt = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp)
    ]

    if filt.empty:
        return None

    year_data = filt[filt["work_year"] == year]["salary_in_usd"]

    if year_data.empty:
        return None

    return year_data.mean()


# ------------------------------------------------------------
# MAIN FUNCTION: get_salary()
# ------------------------------------------------------------
def get_salary(year, job, exp, emp_type, size, method):

    # ----------------------------------------------------
    # 1. RADIUS ML MODEL (No growth logic)
    # ----------------------------------------------------
    if method == "Radius":
        input_row = pd.DataFrame({
            "work_year": [year],
            "job_title": [job],
            "experience_level": [exp],
            "employment_type": [emp_type],
            "company_size": [size]
        })
        ml_value = radius_ml_model.predict(input_row)[0]
        return max(0, ml_value), "Predicted (Radius ML)"

    # ----------------------------------------------------
    # 2. GROWTH–BASED MODEL (Your Original Logic)
    # ----------------------------------------------------
    profile_history = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["employment_type"] == emp_type) &
        (df["company_size"] == size)
    ]

    # CASE A — Exact year exists
    exact = profile_history[profile_history["work_year"] == year]
    if not exact.empty:
        return exact["salary_in_usd"].iloc[0], "Actual"

    # CASE B — Use similar profiles for this year
    similar = similar_growth(df, job, exp, year)
    if similar is not None:
        return similar, "Estimated from Similar Profiles"

    # CASE C — If nothing exists at all → fallback ML
    fallback_row = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "employment_type": [emp_type],
        "company_size": [size]
    })
    fallback_pred = radius_ml_model.predict(fallback_row)[0]
    return fallback_pred, "Fallback ML"

# ------------------------------------------------------------
# FINAL PREDICTION LOOP
# ------------------------------------------------------------
forecast_data = []

for yr in all_years:
    value, source = get_salary(yr, job, exp, emp_type, size, prediction_method)
    forecast_data.append({
        "Year": yr,
        "Salary (USD)": round(value, 2),
        "Source": source
    })

df_out = pd.DataFrame(forecast_data)

# ------------------------------------------------------------
# DISPLAY OUTPUT
# ------------------------------------------------------------
st.title("Cybersecurity Salary Forecast")
st.write(f"### Selected Role: {job} — {exp}")

st.line_chart(df_out, x="Year", y="Salary (USD)", height=420)

st.dataframe(df_out, use_container_width=True)

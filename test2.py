import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cybersecurity Salary Forecast — ML Model (2020–2030)",
    layout="wide",
    page_icon="🛡️"
)

st.markdown("""
<style>
h1 {
    font-size: 2.4rem !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# =========================================================
# FILTER ONLY PROFILES WITH ACTUAL 2020–2022 DATA
# =========================================================
actual_df = df[df["work_year"].isin([2020, 2021, 2022])]

valid_profiles_df = (
    actual_df.groupby([
        "job_title",
        "experience_level",
        "employment_type",
        "company_size",
        "company_location"
    ])
    .size()
    .reset_index()
)

# =========================================================
# SIDEBAR — PROFILE SELECTION
# =========================================================
st.sidebar.header("Select Profile for Prediction")

job_title = st.sidebar.selectbox("Job Title", sorted(valid_profiles_df["job_title"].unique()))
exp_level = st.sidebar.selectbox("Experience Level", sorted(valid_profiles_df["experience_level"].unique()))
emp_type = st.sidebar.selectbox("Employment Type", sorted(valid_profiles_df["employment_type"].unique()))
company_size = st.sidebar.selectbox("Company Size", sorted(valid_profiles_df["company_size"].unique()))
company_loc = st.sidebar.selectbox("Company Location", sorted(valid_profiles_df["company_location"].unique()))

model_choice = st.sidebar.selectbox(
    "Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

# =========================================================
# TARGET + FEATURES
# =========================================================
FEATURES = ["work_year", "job_title", "experience_level", "employment_type",
            "company_size", "company_location"]

TARGET = "salary_in_usd"

# =========================================================
# TRAIN ML MODELS
# =========================================================
@st.cache_resource
def train_models(df):

    X = df[FEATURES]
    y = df[TARGET]

    cat_cols = ["job_title", "experience_level", "employment_type",
                "company_size", "company_location"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
    }

    trained = {}
    for name, est in models.items():
        pipe = Pipeline([
            ("prep", preprocess),
            ("model", est)
        ])
        pipe.fit(X, y)
        trained[name] = pipe

    return trained

trained_models = train_models(df)
model = trained_models[model_choice]

# =========================================================
# FORECAST FUNCTION (PURE ML)
# =========================================================
def predict_salary(year):
    profile = pd.DataFrame([{
        "work_year": year,
        "job_title": job_title,
        "experience_level": exp_level,
        "employment_type": emp_type,
        "company_size": company_size,
        "company_location": company_loc
    }])

    predicted = model.predict(profile)[0]
    return max(0, predicted)

# =========================================================
# CHECK IF PROFILE HAS ACTUAL DATA
# =========================================================
profile_actual = actual_df[
    (actual_df["job_title"] == job_title) &
    (actual_df["experience_level"] == exp_level) &
    (actual_df["employment_type"] == emp_type) &
    (actual_df["company_size"] == company_size) &
    (actual_df["company_location"] == company_loc)
]

# =========================================================
# PAGE TITLE
# =========================================================
st.markdown("# 🛡️ Cybersecurity Salary Forecast — ML Model (2020–2030)")

# If somehow no data exists (should not happen due to filtering)
if profile_actual.empty:
    st.error("This profile has **no actual data** in 2020–2022. (This should not happen.)")
    st.stop()

# =========================================================
# BUILD FORECAST TABLE
# =========================================================
years = list(range(2020, 2031))
forecast_rows = []

for yr in years:

    # 2020–2022 → use actual data
    if yr in [2020, 2021, 2022]:
        actual_row = profile_actual[profile_actual["work_year"] == yr]

        if not actual_row.empty:
            salary = actual_row["salary_in_usd"].mean()
            source = "Actual"
        else:
            salary = None
            source = "No Data"

    else:
        # 2023–2030 → always ML
        salary = predict_salary(yr)
        source = "ML Predicted"

    forecast_rows.append({
        "year": yr,
        "salary": salary,
        "source": source
    })

forecast_df = pd.DataFrame(forecast_rows)

# =========================================================
# WARNING IF MISSING ACTUAL YEARS
# =========================================================
missing_actual = forecast_df[
    (forecast_df["year"].isin([2020, 2021, 2022])) &
    (forecast_df["source"] == "No Data")
]

if len(missing_actual) > 0:
    st.warning("⚠️ Some actual years (2020–2022) are missing for this profile.")

# =========================================================
# PLOT — Actual + Predicted
# =========================================================
fig = go.Figure()

# Actual
actual_pts = forecast_df[forecast_df["source"] == "Actual"]
fig.add_trace(go.Scatter(
    x=actual_pts["year"],
    y=actual_pts["salary"],
    mode="lines+markers",
    name="Actual (2020–2022)",
    line=dict(color="cyan", width=4),
    marker=dict(size=10)
))

# Predicted
pred_pts = forecast_df[forecast_df["source"] == "ML Predicted"]
fig.add_trace(go.Scatter(
    x=pred_pts["year"],
    y=pred_pts["salary"],
    mode="lines+markers",
    name="ML Predicted",
    line=dict(color="orange", width=4, dash="dash"),
    marker=dict(size=10)
))

# No-data markers
none_pts = forecast_df[forecast_df["source"] == "No Data"]
fig.add_trace(go.Scatter(
    x=none_pts["year"],
    y=none_pts["salary"],
    mode="markers",
    name="No Data",
    marker=dict(color="gray", size=12, symbol="x")
))

fig.update_layout(
    height=500,
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    title=f"Salary Forecast (2020–2030)",
    template="plotly_dark",
    legend=dict(orientation="h", y=1.1)
)

st.plotly_chart(fig, width="stretch")

# =========================================================
# SHOW TABLE
# =========================================================
st.markdown("## Forecast Data")
st.dataframe(forecast_df, height=400)

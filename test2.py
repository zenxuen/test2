import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ================================================
# PAGE CONFIG
# ================================================
st.set_page_config(
    page_title="Cybersecurity Salary Forecast (2023–2030)",
    page_icon="🛡️",
    layout="wide"
)

# ================================================
# LOAD DATA
# ================================================
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# =========================================================
# TRAIN ML MODEL (Predict salary_in_usd)
# =========================================================
feature_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "work_year"
]

# Target
target_col = "salary_usd"

X = df[feature_cols]
y = df[target_col]

# Categorical columns
cat_cols = ["experience_level", "employment_type", "job_title", "company_size"]
num_cols = ["work_year"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = Pipeline([
    ("prep", preprocessor),
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

model.fit(X, y)

# =========================================================
# JOB ROLE AUTO-DETECTION FOR GROWTH ADJUSTMENT
# =========================================================
high_growth_keywords = [
    "cloud", "application", "devsec", "security engineer", "architect",
    "pentest", "pen test", "offensive", "red team", "crypto",
    "blockchain", "threat"
]

medium_growth_keywords = [
    "soc", "analyst", "incident", "response",
    "forensic", "vulnerability", "grc", "governance"
]

def detect_role_growth(job_title):
    title = job_title.lower()

    if any(k in title for k in high_growth_keywords):
        return 1.012   # +1.2%
    elif any(k in title for k in medium_growth_keywords):
        return 1.005   # +0.5%
    else:
        return 1.000   # no extra boost

# =========================================================
# INFLATION TABLE 2023–2030
# =========================================================
inflation_table = {
    2023: 0.065,
    2024: 0.032,
    2025: 0.026,
    2026: 0.023,
    2027: 0.021,
    2028: 0.020,
    2029: 0.020,
    2030: 0.020,
}

# =========================================================
# STREAMLIT UI — SIDEBAR
# =========================================================
st.sidebar.title("🎛️ Input Profile")

job_titles = sorted(df["job_title"].unique())
exp_levels = sorted(df["experience_level"].unique())
emp_types = sorted(df["employment_type"].unique())
company_sizes = sorted(df["company_size"].unique())

selected_job = st.sidebar.selectbox("Job Title", job_titles)
selected_exp = st.sidebar.selectbox("Experience Level", exp_levels)
selected_emp = st.sidebar.selectbox("Employment Type", emp_types)
selected_size = st.sidebar.selectbox("Company Size", company_sizes)

start_year, end_year = st.sidebar.slider(
    "Forecast Year Range",
    2023, 2030, (2023, 2030)
)

# =========================================================
# APPLY ML FORECAST
# =========================================================
years = list(range(start_year, end_year + 1))

# Predict using ML for each year
pred_data = pd.DataFrame({
    "experience_level": selected_exp,
    "employment_type": selected_emp,
    "job_title": selected_job,
    "company_size": selected_size,
    "work_year": years
})

ml_pred = model.predict(pred_data)

# =========================================================
# APPLY INFLATION + ROLE ADJUSTMENT
# =========================================================
role_factor = detect_role_growth(selected_job)

adjusted_pred = []
current_value = ml_pred[0]

for i, yr in enumerate(years):
    infl = inflation_table.get(yr, 0.02)  # default 2%

    if i == 0:
        current_value = ml_pred[0]
    else:
        current_value *= (1 + infl) * role_factor

    adjusted_pred.append(current_value)

# =========================================================
# MAIN UI OUTPUT
# =========================================================
st.title("🛡️ Cybersecurity Salary Forecast (2023–2030)")
st.markdown("#### Machine Learning Engine: XGBoost + Role-Aware Inflation Adjustment")

st.subheader(f"📊 Forecast for: **{selected_job}**")

chart_df = pd.DataFrame({
    "year": years,
    "pred_salary": adjusted_pred
})

st.line_chart(chart_df, x="year", y="pred_salary", height=380)

# Show raw data
with st.expander("📄 Raw Prediction Data"):
    st.write(chart_df)


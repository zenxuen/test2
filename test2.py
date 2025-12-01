import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cybersecurity Salary Forecast (ML Powered)",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------
required_cols = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "salary_in_usd",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
]

df = df[required_cols].dropna()


# ---------------------------------------------------------
# BUILD ML MODEL (XGBOOST + TARGET ENCODING)
# ---------------------------------------------------------

# features used internally (even if UI does not expose them)
feature_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "work_year",
]

target_col = "salary_in_usd"

# separate X and y
X = df[feature_cols]
y = df[target_col]

# define which columns are categorical
categorical_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "employee_residence",
    "company_location",
]

numeric_cols = ["remote_ratio", "work_year"]


# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("te", TargetEncoder(), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# define model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
)

# build pipeline
model = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", xgb_model),
    ]
)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train
model.fit(X_train, y_train)


# ---------------------------------------------------------
# SIDEBAR UI (simple!)
# ---------------------------------------------------------

st.sidebar.header("🔧 Input Profile")

job_list = sorted(df["job_title"].unique())
exp_list = sorted(df["experience_level"].unique())
emp_list = sorted(df["employment_type"].unique())
size_list = sorted(df["company_size"].unique())

ui_job = st.sidebar.selectbox("Job Title", job_list)
ui_exp = st.sidebar.selectbox("Experience Level", exp_list)
ui_emp = st.sidebar.selectbox("Employment Type", emp_list)
ui_size = st.sidebar.selectbox("Company Size", size_list)

year_range = st.sidebar.slider(
    "Forecast Year Range",
    min_value=2023,
    max_value=2030,
    value=(2023, 2030),
    step=1
)


# ---------------------------------------------------------
# INTERNAL FEATURE AUTO-FILLING LOGIC
# ---------------------------------------------------------
def autofill_value(col, job):
    """Auto-fill missing ML features using statistical values."""
    subset = df[df["job_title"] == job]
    if len(subset) > 3:
        return subset[col].mode()[0]
    return df[col].mode()[0]


def build_feature_row(year):
    """Construct ML input row using UI selections + auto-filled backend features."""
    return {
        "experience_level": ui_exp,
        "employment_type": ui_emp,
        "job_title": ui_job,
        "company_size": ui_size,

        # Auto-fill internal features
        "employee_residence": autofill_value("employee_residence", ui_job),
        "company_location": autofill_value("company_location", ui_job),
        "remote_ratio": float(autofill_value("remote_ratio", ui_job)),

        # year for trend learning
        "work_year": int(year),
    }


# ---------------------------------------------------------
# GENERATE FORECAST
# ---------------------------------------------------------
start_year, end_year = year_range
years = list(range(start_year, end_year + 1))

prediction_rows = pd.DataFrame([build_feature_row(y) for y in years])
predictions = model.predict(prediction_rows)

forecast_df = pd.DataFrame({
    "year": years,
    "pred_salary": predictions
})


# ---------------------------------------------------------
# DISPLAY OUTPUT
# ---------------------------------------------------------
st.title("🛡️ Cybersecurity Salary Forecast (2023–2030)")
st.write("**Machine Learning Engine: XGBoost + Target Encoding**")

st.subheader(f"📈 Forecast for: {ui_job}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_df["year"],
    y=forecast_df["pred_salary"],
    mode="lines+markers",
    line=dict(width=3, color="#4CAF50"),
    marker=dict(size=8),
    name="Predicted Salary"
))

fig.update_layout(
    title="Predicted Salary in USD (2023–2030)",
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("📄 Raw Prediction Data")
st.dataframe(forecast_df, use_container_width=True)

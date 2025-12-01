import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cybersecurity Salary Prediction (Pure ML)",
    page_icon="💼",
    layout="wide"
)

st.markdown("<h1 class='main-header'>Cybersecurity Salary Prediction (Pure ML)</h1>", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")

    # Clean columns
    df.columns = df.columns.str.strip()

    # Drop unwanted column
    if "salary_currency" in df.columns:
        df = df.drop(columns=["salary_currency"])

    return df


df = load_data()


# ---------------------------------------------------------
# FEATURE SETUP
# ---------------------------------------------------------
feature_cols = [
    "job_title",
    "experience_level",
    "employment_type",
    "company_size",
    "employee_residence",
    "company_location",
    "remote_ratio"
]

categorical_cols = [
    "job_title",
    "experience_level",
    "employment_type",
    "company_size",
    "employee_residence",
    "company_location"
]

numeric_cols = ["remote_ratio"]

target_col = "salary_in_usd"


# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------
@st.cache_resource
def train_models():

    X = df[feature_cols]
    y = df[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    trained = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        pipe.fit(X, y)
        trained[name] = pipe

    return trained


trained_models = train_models()


# ---------------------------------------------------------
# SIDEBAR — PROFILE SELECTION
# ---------------------------------------------------------
st.sidebar.header("Select Profile for Prediction")

unique_jobs = sorted(df["job_title"].unique())
unique_exp = sorted(df["experience_level"].unique())
unique_emp = sorted(df["employment_type"].unique())
unique_size = sorted(df["company_size"].unique())
unique_res = sorted(df["employee_residence"].unique())
unique_loc = sorted(df["company_location"].unique())
unique_remote = sorted(df["remote_ratio"].unique())

selected_job = st.sidebar.selectbox("Job Title", unique_jobs)
selected_exp = st.sidebar.selectbox("Experience Level", unique_exp)
selected_emp = st.sidebar.selectbox("Employment Type", unique_emp)
selected_size = st.sidebar.selectbox("Company Size", unique_size)
selected_res = st.sidebar.selectbox("Employee Residence", unique_res)
selected_loc = st.sidebar.selectbox("Company Location", unique_loc)
selected_remote = st.sidebar.selectbox("Remote Ratio", unique_remote)

model_type = st.sidebar.selectbox(
    "Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)


# ---------------------------------------------------------
# GENERATE PREDICTION FOR 2023–2030 (FLAT ML VALUE)
# ---------------------------------------------------------
def ml_predict_salary(year):

    pipeline = trained_models[model_type]

    input_row = pd.DataFrame([{
        "job_title": selected_job,
        "experience_level": selected_exp,
        "employment_type": selected_emp,
        "company_size": selected_size,
        "employee_residence": selected_res,
        "company_location": selected_loc,
        "remote_ratio": selected_remote
    }])

    pred = pipeline.predict(input_row)[0]

    return pred


# ---------------------------------------------------------
# BUILD FORECAST TABLE 2020–2030
# ---------------------------------------------------------
forecast = []

for year in range(2020, 2030 + 1):

    if year in [2020, 2021, 2022]:

        row = df[
            (df["job_title"] == selected_job) &
            (df["experience_level"] == selected_exp) &
            (df["employment_type"] == selected_emp) &
            (df["company_size"] == selected_size) &
            (df["employee_residence"] == selected_res) &
            (df["company_location"] == selected_loc) &
            (df["remote_ratio"] == selected_remote) &
            (df["work_year"] == year)
        ]

        if not row.empty:
            forecast.append({
                "year": year,
                "salary": row["salary_in_usd"].mean(),
                "source": "Actual"
            })
            continue

        else:
            forecast.append({
                "year": year,
                "salary": None,
                "source": "No Data"
            })
            continue

    # For 2023–2030 → PURE ML PREDICTION
    salary = ml_predict_salary(year)

    forecast.append({
        "year": year,
        "salary": salary,
        "source": "ML"
    })


forecast_df = pd.DataFrame(forecast)


# ---------------------------------------------------------
# MAIN PLOT
# ---------------------------------------------------------
st.subheader("Salary Forecast (2020–2030)")

fig = go.Figure()

# Actual
actual_df = forecast_df[forecast_df["source"] == "Actual"]
fig.add_trace(go.Scatter(
    x=actual_df["year"],
    y=actual_df["salary"],
    mode="lines+markers",
    name="Actual",
    line=dict(color="blue")
))

# ML predicted
ml_df = forecast_df[forecast_df["source"] == "ML"]
fig.add_trace(go.Scatter(
    x=ml_df["year"],
    y=ml_df["salary"],
    mode="lines+markers",
    name="ML Predicted",
    line=dict(color="orange")
))

# No data
missing_df = forecast_df[forecast_df["source"] == "No Data"]
fig.add_trace(go.Scatter(
    x=missing_df["year"],
    y=missing_df["salary"],
    mode="markers",
    name="No Data",
    marker=dict(color="gray", size=10, symbol="x")
))

fig.update_layout(
    height=500,
    xaxis_title="Year",
    yaxis_title="Salary (USD)"
)

st.plotly_chart(fig, use_container_width=False)


# ---------------------------------------------------------
# SHOW FORECAST TABLE
# ---------------------------------------------------------
st.subheader("Forecast Data")
st.dataframe(forecast_df, width="stretch")


# ---------------------------------------------------------
# FEATURE IMPORTANCE (Tree models only)
# ---------------------------------------------------------
st.subheader("Feature Importance")

pipeline = trained_models[model_type]
model = pipeline.named_steps["model"]

if hasattr(model, "feature_importances_"):

    prep = pipeline.named_steps["prep"]
    ohe = prep.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = cat_feature_names + numeric_cols

    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fig2 = px.bar(imp_df, x="importance", y="feature", orientation="h", width=900, height=500)
    st.plotly_chart(fig2, use_container_width=False)

else:
    st.info("Feature importance is only available for Random Forest and Gradient Boosting.")




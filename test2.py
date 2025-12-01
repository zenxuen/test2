import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Salary Forecast",
    page_icon="💼",
    layout="wide",
)

st.markdown("<h1 style='font-size:3rem;'>💼 ML Salary Forecast (2020–2030)</h1>", unsafe_allow_html=True)
st.markdown("**2020–2022: Actual Data | 2023–2030: ML Predictions**")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("salaries_cyber_clean.csv")

df = load_data()

# Keep required columns only
df = df[[
    "salary_in_usd",
    "job_title",
    "experience_level",
    "employment_type",
    "company_size",
    "work_year"
]].dropna()

# ---------------------------------------------------------
# SIDEBAR - MODEL
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

# ---------------------------------------------------------
# TRAIN ML MODEL (NO YEAR INCLUDED AS FEATURE)
# ---------------------------------------------------------
@st.cache_resource
def train_model(df):
    X = df[["job_title", "experience_level", "employment_type", "company_size"]]
    y = df["salary_in_usd"]

    categorical_cols = ["job_title", "experience_level", "employment_type", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="drop"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    trained = {}
    metrics = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        cv = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=5)

        trained[name] = pipe
        metrics[name] = {
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV Mean": cv.mean(),
            "CV Std": cv.std(),
        }

    return trained, metrics

models, metrics = train_model(df)
selected_model = models[model_choice]

# SIDEBAR METRICS
st.sidebar.subheader("📈 Model Performance")
st.sidebar.metric("R²", f"{metrics[model_choice]['R²']:.3f}")
st.sidebar.metric("MAE", f"${metrics[model_choice]['MAE']:,.0f}")
st.sidebar.metric("RMSE", f"${metrics[model_choice]['RMSE']:,.0f}")

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
st.subheader("🎯 Your Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    job = st.selectbox("Job Title", sorted(df["job_title"].unique()))
with col2:
    exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))
with col3:
    emp = st.selectbox("Employment Type", sorted(df["employment_type"].unique()))
with col4:
    size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

profile = {
    "job_title": job,
    "experience_level": exp,
    "employment_type": emp,
    "company_size": size,
}

# ---------------------------------------------------------
# GENERATE FORECAST 2020–2030
# ---------------------------------------------------------
years = list(range(2020, 2031))
prediction_results = []

for year in years:
    if year <= 2022:
        # actual average salary for this profile
        filtered = df[
            (df["job_title"] == job) &
            (df["experience_level"] == exp) &
            (df["employment_type"] == emp) &
            (df["company_size"] == size) &
            (df["work_year"] == year)
        ]

        if len(filtered) > 0:
            salary = filtered["salary_in_usd"].mean()
            src = "Actual"
        else:
            # If no actual for that year, fallback to ML
            salary = selected_model.predict(pd.DataFrame([profile]))[0]
            src = "ML (No Actual Data)"
    else:
        # ML prediction for 2023–2030
        salary = selected_model.predict(pd.DataFrame([profile]))[0]
        src = "Predicted"

    prediction_results.append({
        "Year": year,
        "Salary": salary,
        "Source": src
    })

forecast_df = pd.DataFrame(prediction_results)

# ---------------------------------------------------------
# DISPLAY NUMBERS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📅 Salary Forecast: 2020 → 2030")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("2020", f"${forecast_df[forecast_df.Year == 2020].Salary.iloc[0]:,.0f}")

with colB:
    st.metric("2025 (Predicted)", f"${forecast_df[forecast_df.Year == 2025].Salary.iloc[0]:,.0f}")

with colC:
    growth = ((forecast_df.Salary.iloc[-1] - forecast_df.Salary.iloc[0]) /
              forecast_df.Salary.iloc[0]) * 100
    st.metric("Total Growth", f"{growth:.1f}%")

# ---------------------------------------------------------
# CHART
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Salary Trend (2020–2030)")

actual = forecast_df[forecast_df["Source"] == "Actual"]
predicted = forecast_df[forecast_df["Source"] != "Actual"]

fig = go.Figure()

# Actual Plot
fig.add_trace(go.Scatter(
    x=actual["Year"],
    y=actual["Salary"],
    mode="lines+markers",
    name="Actual (2020–2022)",
    line=dict(color="green", width=4),
    marker=dict(size=10)
))

# Prediction Plot
fig.add_trace(go.Scatter(
    x=predicted["Year"],
    y=predicted["Salary"],
    mode="lines+markers",
    name="Predicted (2023–2030)",
    line=dict(color="purple", width=4, dash="dot"),
    marker=dict(size=10)
))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TABLE
# ---------------------------------------------------------
with st.expander("📋 View Full Forecast Table"):
    st.dataframe(forecast_df, use_container_width=True)

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

st.title("💼 Salary Prediction Dashboard (Forecast to 2030)")

# ---------------------------------------------------------
# Load Dataset (CSV already in Codespace)
# ---------------------------------------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------
X = df[["work_year", "job_title", "experience_level", "company_size"]]
y = df["salary_in_usd"]

categorical_cols = ["job_title", "experience_level", "company_size"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

model = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

model.fit(X, y)  # ✅ trained model

# ---------------------------------------------------------
# Custom Selection (affects ALL predictions)
# ---------------------------------------------------------
st.subheader("⚙️ Customize Model Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    custom_job = st.selectbox("Job Title", sorted(df["job_title"].unique()))

with col2:
    custom_exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))

with col3:
    custom_size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

# ---------------------------------------------------------
# Forecast 2021–2030 (based on custom selection)
# ---------------------------------------------------------
future_years = np.arange(2021, 2031)  # up to 2030

custom_future_data = pd.DataFrame({
    "work_year": future_years,
    "job_title": [custom_job]*len(future_years),
    "experience_level": [custom_exp]*len(future_years),
    "company_size": [custom_size]*len(future_years)
})

future_predictions = model.predict(custom_future_data)

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Predicted Salary (USD)": future_predictions
})

# ---------------------------------------------------------
# Forecast Graph
# ---------------------------------------------------------
st.subheader(f"📈 Salary Forecast for {custom_job} ({custom_exp}, {custom_size})")

fig = px.line(
    forecast_df,
    x="Year",
    y="Predicted Salary (USD)",

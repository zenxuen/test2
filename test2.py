import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

st.title("💼 Salary Prediction Dashboard (Dynamic & Clear Forecast)")

# ---------------------------------------------------------
# Load Dataset (already in Codespace)
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

model.fit(X, y)

# ---------------------------------------------------------
# Custom Selection
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
# Forecast 2020–2030
# ---------------------------------------------------------
historical_years = df["work_year"].unique()
future_years = np.arange(2021, 2031)

# Historical data for selected job
hist_data = df[
    (df["job_title"] == custom_job) &
    (df["experience_level"] == custom_exp) &
    (df["company_size"] == custom_size)
]

# Forecast data
future_df = pd.DataFrame({
    "work_year": future_years,
    "job_title": custom_job,
    "experience_level": custom_exp,
    "company_size": custom_size
})

future_predictions = model.predict(future_df)

# ---------------------------------------------------------
# Combine for Plotly
# ---------------------------------------------------------
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=hist_data["work_year"],
    y=hist_data["salary_in_usd"],
    mode='lines+markers',
    name='Historical',
    line=dict(color='blue', width=3),
    marker=dict(size=8)
))

# Forecast
fig.add_trace(go.Scatter(
    x=future_years,
    y=future_predictions,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='red', width=4, dash='dash'),
    marker=dict(size=10)
))

fig.update_layout(
    title=f"Salary Forecast for {custom_job} ({custom_exp}, {custom_size})",
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    xaxis=dict(dtick=1),
    hovermode="x unified",
    template="plotly_white"
)

st.subheader("📈 Salary Forecast (2020–2030)")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Single Year Prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Salary for a Specific Year")
single_year = st.slider("Select Year", 2020, 2030, 2023)
single_input = pd.DataFrame({
    "work_year": [single_year],
    "job_title": [custom_job],
    "experience_level": [custom_exp],
    "company_size": [custom_size]
})
single_prediction = model.predict(single_input)[0]
st.metric("💰 Predicted Salary", f"${single_prediction:,.2f}")

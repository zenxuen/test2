import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="💼 Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

st.title("💼 Salary Prediction Dashboard (Pro Version)")

# -----------------------------
# Load Dataset (CSV already in Codespace)
# -----------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

# -----------------------------
# Model Training (memorise dataset)
# -----------------------------
X = df[["work_year", "job_title", "experience_level", "company_size"]]
y = df["salary_in_usd"]

categorical_cols = ["job_title", "experience_level", "company_size"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)],
    remainder="passthrough"
)

model = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

model.fit(X, y)

# -----------------------------
# Custom Selection
# -----------------------------
st.subheader("⚙️ Customize Model Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    custom_job = st.selectbox("Job Title", sorted(df["job_title"].unique()))

with col2:
    custom_exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))

with col3:
    custom_size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

# -----------------------------
# Forecast 2021–2035
# -----------------------------
future_years = np.arange(2021, 2036)
custom_future_data = pd.DataFrame({
    "work_year": future_years,
    "job_title": custom_job,
    "experience_level": custom_exp,
    "company_size": custom_size
})

future_predictions = model.predict(custom_future_data)

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Predicted Salary (USD)": future_predictions
})

# -----------------------------
# Interactive Forecast Graph
# -----------------------------
st.subheader("📈 Salary Forecast (2021–2035)")

fig = go.Figure()

# Plot memorised dataset as scatter for context
fig.add_trace(go.Scatter(
    x=df["work_year"],
    y=df["salary_in_usd"],
    mode="markers",
    name="Actual Dataset",
    marker=dict(size=8, color="gray", opacity=0.5),
    hovertemplate="Year: %{x}<br>Salary: $%{y:,.0f}<extra></extra>"
))

# Plot predicted line for custom selection
fig.add_trace(go.Scatter(
    x=forecast_df["Year"],
    y=forecast_df["Predicted Salary (USD)"],
    mode="lines+markers",
    name=f"{custom_job} ({custom_exp}, {custom_size})",
    line=dict(width=4, dash="dash"),
    marker=dict(size=10),
    hovertemplate="Year: %{x}<br>Predicted Salary: $%{y:,.0f}<extra></extra>"
))

fig.update_layout(
    template="plotly_white",
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    xaxis=dict(dtick=1),
    hovermode="x unified",
    title="Salary Forecast vs Actual Data"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Single Year Prediction
# -----------------------------
st.subheader("🔮 Predict Salary for a Specific Year")
single_year = st.slider("Select Year", 2020, 2035, 2023)

single_input = pd.DataFrame({
    "work_year": [single_year],
    "job_title": [custom_job],
    "experience_level": [custom_exp],
    "company_size": [custom_size]
})

single_prediction = model.predict(single_input)[0]
st.metric("💰 Predicted Salary", f"${single_prediction:,.2f}")

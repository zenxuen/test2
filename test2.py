import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

st.title("💼 Salary Prediction Dashboard (Dynamic Model)")

# ---------------------------------------------------------
# Load Dataset (Fixed Path)
# ---------------------------------------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

st.write("✅ Dataset loaded successfully!")

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
# Filter Dataset Based on Selection
# ---------------------------------------------------------
filtered_df = df[
    (df["job_title"] == custom_job) &
    (df["experience_level"] == custom_exp) &
    (df["company_size"] == custom_size)
]

if len(filtered_df) < 2:
    st.warning("Not enough data for this selection. Using overall dataset instead.")
    filtered_df = df

# ---------------------------------------------------------
# Train Model (Filtered)
# ---------------------------------------------------------
X = filtered_df[["work_year"]]
y = filtered_df["salary_in_usd"]

model = LinearRegression()
model.fit(X, y)

# ---------------------------------------------------------
# Forecast 2021–2025
# ---------------------------------------------------------
future_years = np.arange(2021, 2026).reshape(-1, 1)
future_predictions = model.predict(future_years)

forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Salary (USD)": future_predictions
})

# ---------------------------------------------------------
# Forecast Graph
# ---------------------------------------------------------
st.subheader("📈 Salary Forecast Based on Your Selections (2021–2025)")

fig = px.line(
    forecast_df,
    x="Year",
    y="Predicted Salary (USD)",
    markers=True,
    title=f"Salary Forecast for {custom_job} ({custom_exp}, {custom_size})",
    template="plotly_white"
)

# Increase gap between markers and thicker line
fig.update_traces(line=dict(width=3, dash='dot'), marker=dict(size=10))

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Single Year Custom Prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Salary for a Specific Year")

single_year = st.slider("Select Year", 2020, 2035, 2023)
single_prediction = model.predict(np.array([[single_year]]))[0]

st.metric("💰 Predicted Salary", f"${single_prediction:,.2f}")

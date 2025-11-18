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

st.title("💼 Salary Prediction Dashboard (Dynamic Model)")

# ---------------------------------------------------------
# Load Dataset (CSV already in Codespace)
# ---------------------------------------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# Features and Target
# ---------------------------------------------------------
X = df[["work_year", "job_title", "experience_level", "company_size"]]
y = df["salary_in_usd"]

categorical_cols = ["job_title", "experience_level", "company_size"]
numerical_cols = ["work_year"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

# Fit the model on the full dataset
model.fit(X, y)

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
# Forecast 2021–2035 (based on custom selection)
# ---------------------------------------------------------
future_years = np.arange(2021, 2036)
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
st.subheader("📈 Salary Forecast Based on Your Selections (2021–2035)")

fig = px.line(
    forecast_df,
    x="Year",
    y="Predicted Salary (USD)",
    markers=True,
    title=f"Salary Forecast for {custom_job} ({custom_exp}, {custom_size})",
    template="plotly_white"
)
fig.update_traces(line=dict(width=4), marker=dict(size=10))
fig.update_layout(
    yaxis_title="Salary (USD)",
    xaxis=dict(dtick=1),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Single Year Custom Prediction
# ---------------------------------------------------------
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

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

st.title("💼 Salary Prediction Dashboard (Dynamic & Accurate)")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# Train Model with polynomial features for trend
# ---------------------------------------------------------
X = df[["work_year", "job_title", "experience_level", "company_size"]]
y = df["salary_in_usd"]

categorical_cols = ["job_title", "experience_level", "company_size"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

poly = PolynomialFeatures(degree=2, include_bias=False)

model = Pipeline([
    ("prep", preprocessor),
    ("poly", poly),
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
# Separate historical and future years
# ---------------------------------------------------------
historical_data = df[
    (df["job_title"] == custom_job) &
    (df["experience_level"] == custom_exp) &
    (df["company_size"] == custom_size)
]

last_historical_year = historical_data["work_year"].max()
future_years = np.arange(last_historical_year + 1, 2031)  # forecast starts after last historical year

# Future predictions
if len(future_years) > 0:
    future_df = pd.DataFrame({
        "work_year": future_years,
        "job_title": custom_job,
        "experience_level": custom_exp,
        "company_size": custom_size
    })
    future_predictions = model.predict(future_df)
else:
    future_df = pd.DataFrame(columns=["work_year"])
    future_predictions = []

# ---------------------------------------------------------
# Plotly Forecast Graph
# ---------------------------------------------------------
fig = go.Figure()

# Historical actual data
fig.add_trace(go.Scatter(
    x=historical_data["work_year"],
    y=historical_data["salary_in_usd"],
    mode='lines+markers',
    name='Historical',
    line=dict(color='blue', width=3),
    marker=dict(size=8)
))

# Future forecast
if len(future_years) > 0:
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

st.subheader("📈 Salary Forecast (Historical + Forecast)")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Single Year Prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Salary for a Specific Year")

single_year = st.slider("Select Year", int(df["work_year"].min()), 2030, int(df["work_year"].min()))

if single_year in historical_data["work_year"].values:
    single_prediction = historical_data.loc[historical_data["work_year"] == single_year, "salary_in_usd"].values[0]
else:
    single_input = pd.DataFrame({
        "work_year": [single_year],
        "job_title": [custom_job],
        "experience_level": [custom_exp],
        "company_size": [custom_size]
    })
    single_prediction = model.predict(single_input)[0]

st.metric("💰 Predicted Salary", f"${single_prediction:,.2f}")

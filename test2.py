import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
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

st.title("💼 Salary Prediction Dashboard (Dynamic Model with Historical Accuracy)")

# ---------------------------------------------------------
# Load Dataset
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

model.fit(X, y)  # Model is trained and memorizes the dataset

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
].sort_values("work_year")

last_year = historical_data["work_year"].max()
future_years = np.arange(last_year + 1, 2031)

if len(future_years) > 0:
    future_data = pd.DataFrame({
        "work_year": future_years,
        "job_title": custom_job,
        "experience_level": custom_exp,
        "company_size": custom_size
    })
    future_predictions = model.predict(future_data)
else:
    future_predictions = np.array([])

# ---------------------------------------------------------
# Prepare combined data for plotting
# ---------------------------------------------------------
forecast_df = pd.DataFrame({
    "Year": list(historical_data["work_year"]) + list(future_years),
    "Salary (USD)": list(historical_data["salary_in_usd"]) + list(future_predictions),
    "Type": ["Actual"]*len(historical_data) + ["Forecast"]*len(future_years)
})

# ---------------------------------------------------------
# Forecast Graph
# ---------------------------------------------------------
st.subheader("📈 Salary Forecast with Historical Accuracy")

fig = go.Figure()

# Historical actual
fig.add_trace(go.Scatter(
    x=forecast_df[forecast_df["Type"]=="Actual"]["Year"],
    y=forecast_df[forecast_df["Type"]=="Actual"]["Salary (USD)"],
    mode="lines+markers",
    name="Actual",
    line=dict(color="blue", width=3),
    marker=dict(size=8)
))

# Forecast
if len(future_years) > 0:
    fig.add_trace(go.Scatter(
        x=forecast_df[forecast_df["Type"]=="Forecast"]["Year"],
        y=forecast_df[forecast_df["Type"]=="Forecast"]["Salary (USD)"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="red", width=3, dash="dash"),
        marker=dict(size=8)
    ))
    fig.add_vrect(
        x0=future_years[0]-0.5, x1=future_years[-1]+0.5,
        fillcolor="lightpink", opacity=0.2,
        layer="below", line_width=0
    )

fig.update_layout(
    xaxis=dict(dtick=1),
    yaxis_title="Salary (USD)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Single Year Custom Prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Salary for a Specific Year")

single_year = st.slider("Select Year", 2020, 2030, 2023)

if single_year in historical_data["work_year"].values:
    single_prediction = historical_data[historical_data["work_year"]==single_year]["salary_in_usd"].values[0]
else:
    single_input = pd.DataFrame({
        "work_year": [single_year],
        "job_title": [custom_job],
        "experience_level": [custom_exp],
        "company_size": [custom_size]
    })
    single_prediction = model.predict(single_input)[0]

st.metric("💰 Predicted Salary", f"${single_prediction:,.2f}")

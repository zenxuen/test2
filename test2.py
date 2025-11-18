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

st.title("💼 Salary Prediction Dashboard (Enhanced Version)")

# ---------------------------------------------------------
# Load Dataset (CSV inside Codespace)
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
# Model Inputs (Dynamic Selection)
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
# Forecast 2021–2035
# ---------------------------------------------------------
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

# Smooth the line visually (optional but better looking)
forecast_df["Smoothed"] = forecast_df["Predicted Salary (USD)"].rolling(3, min_periods=1).mean()

# ---------------------------------------------------------
# Forecast Graph (Improved Visibility and Gaps)
# ---------------------------------------------------------
st.subheader("📈 Salary Forecast Based on Your Selections (2021–2035)")

fig = px.line(
    forecast_df,
    x="Year",
    y="Smoothed",
    markers=True,
    title=f"Salary Forecast for {custom_job} ({custom_exp}, {custom_size})",
    template="plotly_white"
)

# Improved visuals
fig.update_traces(
    line=dict(width=5),
    marker=dict(size=11)
)

fig.update_layout(
    yaxis_title="Salary (USD)",
    xaxis=dict(
        dtick=1,
        tickangle=45,
        tickfont=dict(size=13)
    ),
    yaxis=dict(tickfont=dict(size=13)),
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Single Year Predictor
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

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

model.fit(X, y)

# ---------------------------------------------------------
# Custom Selection (affects future predictions only)
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
# Forecast future years 2023–2030 (historical years remain fixed)
# ---------------------------------------------------------
historical_years = df["work_year"].values
historical_salaries = df["salary_in_usd"].values

future_years = np.arange(max(historical_years)+1, 2031)  # 2023–2030

if len(future_years) > 0:
    future_data = pd.DataFrame({
        "work_year": future_years,
        "job_title": custom_job,
        "experience_level": custom_exp,
        "company_size": custom_size
    })
    future_predictions = model.predict(future_data)
else:
    future_years = np.array([])
    future_predictions = np.array([])

# Combine historical and future data
all_years = np.concatenate([historical_years, future_years])
all_salaries = np.concatenate([historical_salaries, future_predictions])

forecast_df = pd.DataFrame({
    "Year": all_years,
    "Salary (USD)": all_salaries
})

# ---------------------------------------------------------
# Forecast Graph
# ---------------------------------------------------------
st.subheader("📈 Salary Forecast (Historical + Predicted)")

fig = px.line(
    forecast_df,
    x="Year",
    y="Salary (USD)",
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

single_year = st.slider("Select Year", 2020, 2030, 2023)

if single_year in historical_years:
    single_prediction = df.loc[df["work_year"] == single_year, "salary_in_usd"].values[0]
else:
    single_input = pd.DataFrame({
        "work_year": [single_year],
        "job_title": [custom_job],
        "experience_level": [custom_exp],
        "company_size": [custom_size]
    })
    single_prediction = model.predict(single_input)[0]

st.metric("💰 Predicted Salary", f"${single_prediction:,.2f}")

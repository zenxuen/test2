import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Cybersecurity Salary Forecast (2020–2030)",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cybersecurity Salary Forecast — ML Model (2020–2030)")

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")     # ← replace with your name
    return df

df = load_data()

# -------------------------------------------------
# Filter only profiles that actually exist
# -------------------------------------------------
def get_valid_profiles(df):
    combo = df.groupby([
        "job_title",
        "experience_level",
        "employment_type",
        "company_size",
        "company_location"
    ]).size().reset_index()

    return combo

valid_profiles_df = get_valid_profiles(df)

# Sidebar selectors
st.sidebar.header("🔎 Select Profile")

job = st.sidebar.selectbox("Job Title", sorted(valid_profiles_df["job_title"].unique()))
exp = st.sidebar.selectbox("Experience Level", sorted(valid_profiles_df["experience_level"].unique()))
emp = st.sidebar.selectbox("Employment Type", sorted(valid_profiles_df["employment_type"].unique()))
size = st.sidebar.selectbox("Company Size", sorted(valid_profiles_df["company_size"].unique()))
loc = st.sidebar.selectbox("Company Location", sorted(valid_profiles_df["company_location"].unique()))

# Filter dataset for this profile
profile_df = df[
    (df["job_title"] == job) &
    (df["experience_level"] == exp) &
    (df["employment_type"] == emp) &
    (df["company_size"] == size) &
    (df["company_location"] == loc)
]

if profile_df.empty:
    st.error("This profile has no actual data in your dataset. It should not happen because we filtered it out.")
    st.stop()

# -------------------------------------------------
# Train ML Model (Random Forest)
# -------------------------------------------------

FEATURES = ["job_title", "experience_level", "employment_type",
            "company_size", "company_location", "work_year"]

TARGET = "salary_in_usd"

X = df[FEATURES]
y = df[TARGET]

categorical_cols = ["job_title", "experience_level", "employment_type",
                    "company_size", "company_location"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", ["work_year"])
    ]
)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None
    ))
])

model.fit(X, y)

# -------------------------------------------------
# Build Forecast Table
# -------------------------------------------------

def predict_salary_for_year(year):
    data = pd.DataFrame([{
        "job_title": job,
        "experience_level": exp,
        "employment_type": emp,
        "company_size": size,
        "company_location": loc,
        "work_year": year
    }])

    pred = model.predict(data)[0]
    return round(pred, 2)

forecast_rows = []

# Actual years: 2020–2022
for yr in [2020, 2021, 2022]:
    subset = profile_df[profile_df["work_year"] == yr]

    if subset.empty:
        forecast_rows.append({
            "year": yr,
            "salary": None,
            "source": "No Data"
        })
    else:
        avg = subset["salary_in_usd"].mean()
        forecast_rows.append({
            "year": yr,
            "salary": round(avg, 2),
            "source": "Actual"
        })

# Predicted years 2023–2030
for yr in range(2023, 2030 + 1):
    forecast_rows.append({
        "year": yr,
        "salary": predict_salary_for_year(yr),
        "source": "Predicted"
    })

forecast_df = pd.DataFrame(forecast_rows)

# -------------------------------------------------
# Display Table
# -------------------------------------------------
st.subheader("📘 Salary Forecast Table (2020–2030)")
st.dataframe(forecast_df, use_container_width=True)

# -------------------------------------------------
# Plot Trend (Plotly)
# -------------------------------------------------
st.subheader("📈 Salary Trend Chart")

fig = go.Figure()

# Actual points
fig.add_trace(go.Scatter(
    x=forecast_df[forecast_df["source"] == "Actual"]["year"],
    y=forecast_df[forecast_df["source"] == "Actual"]["salary"],
    mode="markers+lines",
    name="Actual",
    line=dict(color="green", width=3)
))

# Predicted points
fig.add_trace(go.Scatter(
    x=forecast_df[forecast_df["source"] == "Predicted"]["year"],
    y=forecast_df[forecast_df["source"] == "Predicted"]["salary"],
    mode="markers+lines",
    name="Predicted",
    line=dict(color="blue", width=3, dash="dash")
))

fig.update_layout(
    height=500,
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    legend_title="Source"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------

st.subheader("🛠 Feature Importance (Random Forest)")

rf_model = model.named_steps["model"]
prep = model.named_steps["prep"]

# Get feature names from encoder
encoded_cols = list(prep.named_transformers_["cat"].get_feature_names_out(categorical_cols))
final_features = encoded_cols + ["work_year"]

importances = rf_model.feature_importances_

imp_df = pd.DataFrame({
    "feature": final_features,
    "importance": importances
}).sort_values("importance", ascending=False)

fig2 = go.Figure(go.Bar(
    x=imp_df["importance"],
    y=imp_df["feature"],
    orientation="h"
))

fig2.update_layout(
    height=600,
    xaxis_title="Importance Score",
    yaxis_title="Feature"
)

st.plotly_chart(fig2, use_container_width=True)



import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cybersecurity Salary Forecast (Hybrid ML)",
    page_icon="💼",
    layout="wide",
)

st.markdown(
    "<h1 style='font-size:2.5rem;'>2020–2022 Actual | 2023–2030 Hybrid ML Predictions</h1>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("salaries_cyber_clean.csv")

df = load_data()

df = df[[
    "salary_in_usd",
    "work_year",
    "job_title",
    "experience_level",
    "employment_type",
    "company_size",
]].dropna()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Hybrid Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose Profile Model",
    ["Random Forest", "Gradient Boosting"]
)

# ---------------------------------------------------------
# MODEL 1 – TIME TREND (LR)
# ---------------------------------------------------------
@st.cache_resource
def train_time_model(data):
    X = data[["work_year"]]
    y = data["salary_in_usd"]

    lr = LinearRegression()
    lr.fit(X, y)

    return lr

time_model = train_time_model(df)

# ---------------------------------------------------------
# MODEL 2 – PROFILE MODEL (RF or GB)
# ---------------------------------------------------------
@st.cache_resource
def train_profile_model(data):

    X = data[["job_title", "experience_level", "employment_type", "company_size"]]
    y = data["salary_in_usd"]

    cat_cols = ["job_title", "experience_level", "employment_type", "company_size"]

    preproc = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    if model_choice == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=250, random_state=42, n_jobs=-1,
            max_depth=None, min_samples_split=3
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=250, random_state=42,
            learning_rate=0.05, max_depth=3
        )

    pipe = Pipeline([
        ("prep", preproc),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

    metrics = {
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "CV Mean": cv.mean(),
        "CV Std": cv.std()
    }

    return pipe, metrics

profile_model, pmetrics = train_profile_model(df)

# ---------------------------------------------------------
# DISPLAY MODEL METRICS
# ---------------------------------------------------------
st.sidebar.subheader("📈 Profile Model Performance")

st.sidebar.metric("R²", f"{pmetrics['R²']:.3f}")
st.sidebar.metric("MAE", f"${pmetrics['MAE']:,.0f}")
st.sidebar.metric("RMSE", f"${pmetrics['RMSE']:,.0f}")
st.sidebar.caption(f"CV R² = {pmetrics['CV Mean']:.3f} ± {pmetrics['CV Std']:.3f}")

# ---------------------------------------------------------
# USER PROFILE INPUT
# ---------------------------------------------------------
st.subheader("🎯 Your Profile")

c1, c2, c3, c4 = st.columns(4)
job = c1.selectbox("Job Title", sorted(df["job_title"].unique()))
exp = c2.selectbox("Experience Level", sorted(df["experience_level"].unique()))
emp = c3.selectbox("Employment Type", sorted(df["employment_type"].unique()))
size = c4.selectbox("Company Size", sorted(df["company_size"].unique()))

profile = {
    "job_title": job,
    "experience_level": exp,
    "employment_type": emp,
    "company_size": size,
}

dataset_mean = df["salary_in_usd"].mean()

# ---------------------------------------------------------
# STRICT FORECAST (2020–2022 actual | 2023–2030 ML ONLY)
# ---------------------------------------------------------
years = list(range(2020, 2031))
rows = []

for year in years:

    # ===== A) 2020–2022 MUST BE ACTUAL ONLY =====
    if year <= 2022:
        subset = df[
            (df["work_year"] == year) &
            (df["job_title"] == job) &
            (df["experience_level"] == exp) &
            (df["employment_type"] == emp) &
            (df["company_size"] == size)
        ]

        if len(subset) > 0:
            salary = subset["salary_in_usd"].mean()
            source = "Actual"
        else:
            salary = np.nan
            source = "No Data"

    # ===== B) 2023–2030 MUST BE HYBRID PREDICTION =====
    else:
        trend = time_model.predict([[year]])[0]
        base = profile_model.predict(pd.DataFrame([profile]))[0]
        salary = trend + (base - dataset_mean)
        source = "Predicted (Hybrid)"

    rows.append({
        "Year": year,
        "Salary": salary,
        "Source": source
    })

forecast_df = pd.DataFrame(rows)

# ---------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📅 Salary Forecast: 2020 → 2030")

start = forecast_df.loc[forecast_df["Year"] == 2020, "Salary"].iloc[0]
end = forecast_df.loc[forecast_df["Year"] == 2030, "Salary"].iloc[0]
mid = forecast_df.loc[forecast_df["Year"] == 2025, "Salary"].iloc[0]

growth = ((end - start) / start) * 100 if not np.isnan(start) else 0

m1, m2, m3 = st.columns(3)
m1.metric("2020", f"${start:,.0f}" if not np.isnan(start) else "N/A")
m2.metric("2025 (Predicted)", f"${mid:,.0f}")
m3.metric("Growth 2020→2030", f"{growth:.1f}%" if not np.isnan(start) else "N/A")

# ---------------------------------------------------------
# CHART
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Salary Trend (Actual vs Hybrid Prediction)")

actual = forecast_df[forecast_df["Source"] == "Actual"]
pred = forecast_df[forecast_df["Source"] != "Actual"]

fig = go.Figure()

if not actual.empty:
    fig.add_trace(go.Scatter(
        x=actual["Year"],
        y=actual["Salary"],
        mode="lines+markers",
        name="Actual (2020–2022)",
        line=dict(color="green", width=4)
    ))

fig.add_trace(go.Scatter(
    x=pred["Year"],
    y=pred["Salary"],
    mode="lines+markers",
    name="Hybrid Predicted (2023–2030)",
    line=dict(color="purple", width=3, dash="dot")
))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    height=500,
    template="plotly_dark",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TABLE
# ---------------------------------------------------------
with st.expander("📋 Forecast Table"):
    df_show = forecast_df.copy()
    df_show["Salary"] = df_show["Salary"].apply(
        lambda x: f"${x:,.0f}" if not np.isnan(x) else "—"
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

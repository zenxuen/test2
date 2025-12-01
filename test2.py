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
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

df = df[[
    "salary_in_usd",
    "work_year",
    "job_title",
    "experience_level",
    "employment_type",
    "company_size"
]].dropna()

# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Hybrid Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose Profile Model",
    ["Random Forest", "Gradient Boosting"]
)

# ---------------------------------------------------------
# TRAIN MODEL 1 — TIME TREND (Linear Regression)
# ---------------------------------------------------------
@st.cache_resource
def train_time_model(df):
    time_df = df[["work_year", "salary_in_usd"]]

    X = time_df[["work_year"]]
    y = time_df["salary_in_usd"]

    lr = LinearRegression()
    lr.fit(X, y)

    return lr

time_model = train_time_model(df)

# ---------------------------------------------------------
# TRAIN MODEL 2 — PROFILE MODEL (RF / GB)
# ---------------------------------------------------------
@st.cache_resource
def train_profile_model(df):

    X = df[["job_title", "experience_level", "employment_type", "company_size"]]
    y = df["salary_in_usd"]

    categorical_cols = ["job_title", "experience_level",
                        "employment_type", "company_size"]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    if model_choice == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            max_depth=None,
            min_samples_split=3,
            n_jobs=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
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


profile_model, profile_metrics = train_profile_model(df)

# ---------------------------------------------------------
# PROFILE INPUT
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
    "company_size": size
}

# ---------------------------------------------------------
# GENERATE FORECAST (Hybrid)
# ---------------------------------------------------------
years = list(range(2020, 2031))
records = []

for year in years:
    # 1. TIME TREND
    trend_value = time_model.predict([[year]])[0]

    # 2. PROFILE BASELINE
    profile_value = profile_model.predict(pd.DataFrame([profile]))[0]

    # Final Salary Prediction
    salary_pred = trend_value + (profile_value - df["salary_in_usd"].mean())

    # For 2020–2022, try actual
    if year <= 2022:
        act = df[
            (df["work_year"] == year) &
            (df["job_title"] == job) &
            (df["experience_level"] == exp) &
            (df["employment_type"] == emp) &
            (df["company_size"] == size)
        ]
        if len(act) > 0:
            final_salary = act["salary_in_usd"].mean()
            source = "Actual"
        else:
            final_salary = salary_pred
            source = "Predicted (Hybrid)"
    else:
        final_salary = salary_pred
        source = "Predicted (Hybrid)"

    records.append({
        "Year": year,
        "Salary": final_salary,
        "Source": source
    })

forecast_df = pd.DataFrame(records)

# ---------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📅 Salary Forecast Summary (2020–2030)")

start = forecast_df.loc[forecast_df.Year == 2020, "Salary"].iloc[0]
mid = forecast_df.loc[forecast_df.Year == 2025, "Salary"].iloc[0]
end = forecast_df.loc[forecast_df.Year == 2030, "Salary"].iloc[0]

growth = ((end - start) / start) * 100

m1, m2, m3 = st.columns(3)
m1.metric("2020", f"${start:,.0f}")
m2.metric("2025 (Hybrid)", f"${mid:,.0f}")
m3.metric("Total Growth 2020–2030", f"{growth:.1f}%")

# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Hybrid Salary Forecast (2020–2030)")

act = forecast_df[forecast_df.Source == "Actual"]
pred = forecast_df[forecast_df.Source != "Actual"]

fig = go.Figure()

if not act.empty:
    fig.add_trace(go.Scatter(
        x=act["Year"],
        y=act["Salary"],
        name="Actual",
        mode="lines+markers",
        line=dict(color="green", width=4)
    ))

fig.add_trace(go.Scatter(
    x=pred["Year"],
    y=pred["Salary"],
    name="Hybrid Predicted",
    mode="lines+markers",
    line=dict(color="purple", width=3, dash="dot")
))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    template="plotly_dark",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TABLE
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📋 Forecast Data Table"):
    df_show = forecast_df.copy()
    df_show["Salary"] = df_show["Salary"].map(lambda x: f"${x:,.0f}")
    st.dataframe(df_show, hide_index=True, use_container_width=True)

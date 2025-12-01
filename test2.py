import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cybersecurity Salary Forecast (ML)",
    page_icon="💼",
    layout="wide",
)

st.markdown(
    "<h1 style='font-size:2.6rem;'>2020–2022: Actual Data | 2023–2030: ML Predictions</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Target: <b>salary_in_usd</b> • Features: <b>work_year, job_title, "
    "experience_level, employment_type, company_size</b>",
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

# Keep only required columns
df = df[[
    "salary_in_usd",
    "work_year",
    "job_title",
    "experience_level",
    "employment_type",
    "company_size"
]].dropna()

# ---------------------------------------------------------
# SIDEBAR – MODEL SETTINGS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ["Random Forest", "Gradient Boosting"]
)

# ---------------------------------------------------------
# TRAIN MODELS (work_year INCLUDED)
# ---------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    X = data[["work_year", "job_title", "experience_level",
              "employment_type", "company_size"]]
    y = data["salary_in_usd"]

    categorical_cols = ["job_title", "experience_level",
                        "employment_type", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"   # keeps work_year as numeric
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    trained = {}
    metrics = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="r2"
        )

        trained[name] = pipe
        metrics[name] = {
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV Mean": cv_scores.mean(),
            "CV Std": cv_scores.std()
        }

    return trained, metrics

models, perf = train_models(df)
selected_model = models[model_choice]

# ---------------------------------------------------------
# SIDEBAR – METRICS
# ---------------------------------------------------------
st.sidebar.subheader("📈 Model Performance")

m = perf[model_choice]
st.sidebar.metric("R²", f"{m['R²']:.3f}")
st.sidebar.metric("MAE", f"${m['MAE']:,.0f}")
st.sidebar.metric("RMSE", f"${m['RMSE']:,.0f}")
st.sidebar.caption(f"CV R² = {m['CV Mean']:.3f} ± {m['CV Std']:.3f}")

# ---------------------------------------------------------
# USER PROFILE INPUT
# ---------------------------------------------------------
st.subheader("🎯 Your Profile")

c1, c2, c3, c4 = st.columns(4)

with c1:
    job = st.selectbox("Job Title", sorted(df["job_title"].unique()))
with c2:
    exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))
with c3:
    emp = st.selectbox("Employment Type", sorted(df["employment_type"].unique()))
with c4:
    size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

profile_base = {
    "job_title": job,
    "experience_level": exp,
    "employment_type": emp,
    "company_size": size,
}

# ---------------------------------------------------------
# BUILD 2020–2030 FORECAST
# ---------------------------------------------------------
years = list(range(2020, 2030 + 1))
records = []

for year in years:
    # try actual data first for 2020–2022
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
            X_row = pd.DataFrame([{
                "work_year": year,
                **profile_base
            }])
            salary = selected_model.predict(X_row)[0]
            source = "Predicted (ML – no actual for this year/profile)"
    else:
        # future years: ML forecast
        X_row = pd.DataFrame([{
            "work_year": year,
            **profile_base
        }])
        salary = selected_model.predict(X_row)[0]
        source = "Predicted (ML)"

    records.append({
        "Year": year,
        "Salary (USD)": salary,
        "Source": source
    })

forecast_df = pd.DataFrame(records)

# ---------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📅 Salary Forecast: 2020 → 2030")

start_salary = forecast_df.loc[forecast_df["Year"] == 2020, "Salary (USD)"].iloc[0]
mid_salary = forecast_df.loc[forecast_df["Year"] == 2025, "Salary (USD)"].iloc[0]
end_salary = forecast_df.loc[forecast_df["Year"] == 2030, "Salary (USD)"].iloc[0]

total_growth_pct = ((end_salary - start_salary) / start_salary) * 100 if start_salary > 0 else 0

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("2020", f"${start_salary:,.0f}")
with m2:
    st.metric("2025 (Predicted)", f"${mid_salary:,.0f}")
with m3:
    st.metric("Total Growth 2020–2030", f"{total_growth_pct:.1f}%")

# ---------------------------------------------------------
# LINE CHART: ACTUAL vs PREDICTED
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Salary Trend (2020–2030)")

actual_df = forecast_df[forecast_df["Source"] == "Actual"]
pred_df = forecast_df[forecast_df["Source"] != "Actual"]

fig = go.Figure()

if not actual_df.empty:
    fig.add_trace(go.Scatter(
        x=actual_df["Year"],
        y=actual_df["Salary (USD)"],
        mode="lines+markers",
        name="Actual (2020–2022)",
        line=dict(color="green", width=4),
        marker=dict(size=10)
    ))

if not pred_df.empty:
    fig.add_trace(go.Scatter(
        x=pred_df["Year"],
        y=pred_df["Salary (USD)"],
        mode="lines+markers",
        name="Predicted (2023–2030 / gaps)",
        line=dict(color="purple", width=4, dash="dot"),
        marker=dict(size=10)
    ))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    template="plotly_white",
    height=550,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TABLE VIEW
# ---------------------------------------------------------
with st.expander("📋 Detailed Forecast Table"):
    df_display = forecast_df.copy()
    df_display["Salary (USD)"] = df_display["Salary (USD)"].map(lambda x: f"${x:,.0f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# SIMPLE EDA (OPTIONAL)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Overall Average Salary by Year (All Profiles)")

year_avg = df.groupby("work_year")["salary_in_usd"].mean().reset_index()

fig2 = px.line(
    year_avg,
    x="work_year",
    y="salary_in_usd",
    markers=True,
    title="Average Salary by Year (Dataset 2020–2022)"
)
fig2.update_layout(xaxis_title="Year", yaxis_title="Avg Salary (USD)")
st.plotly_chart(fig2, use_container_width=True)

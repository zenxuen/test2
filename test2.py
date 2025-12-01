import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💼 Cybersecurity Salary Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("*Actual Data: 2020–2022 | ML Predictions: 2023–2030*")
st.markdown("**Target:** Salary | **Predictors:** Work Year, Job Title, Experience, Company Size")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("salaries_cyber_clean.csv")

df = load_data()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

prediction_method = st.sidebar.radio(
    "Prediction Method",
    ["Growth-Based (Recommended)", "Pure ML Model", "Hybrid"]
)

# ---------------------------------------------------------
# Train Models
# ---------------------------------------------------------
@st.cache_resource
def train_models(data):
    X = data[["work_year", "job_title", "experience_level", "company_size"]]
    y = data["salary_in_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical = ["job_title", "experience_level", "company_size"]
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=7,
            learning_rate=0.1, random_state=42
        )
    }

    trained = {}
    metrics = {}

    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

        trained[name] = pipe
        metrics[name] = {
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV Mean": cv.mean(),
            "CV Std": cv.std()
        }

    return trained, metrics, X_test, y_test

models, performance_metrics, X_test, y_test = train_models(df)
selected_model = models[model_type]

# ---------------------------------------------------------
# VALID PROFILE FILTERING (≥1 actual year)
# ---------------------------------------------------------
valid_profiles = (
    df[df["work_year"].isin([2020, 2021, 2022])]
    .groupby(["job_title", "experience_level", "company_size"])
    .size()
    .reset_index()[["job_title", "experience_level", "company_size"]]
)

# ---------------------------------------------------------
# DROPDOWN 1 — Job Title
# ---------------------------------------------------------
job_options = sorted(valid_profiles["job_title"].unique())
custom_job = st.selectbox("Job Title", job_options)

# ---------------------------------------------------------
# DROPDOWN 2 — Experience Levels filtered by Job
# ---------------------------------------------------------
exp_options = sorted(
    valid_profiles[valid_profiles["job_title"] == custom_job]["experience_level"].unique()
)
custom_exp = st.selectbox("Experience Level", exp_options)

# ---------------------------------------------------------
# DROPDOWN 3 — Company Size filtered by Job + Exp
# ---------------------------------------------------------
size_options = sorted(
    valid_profiles[
        (valid_profiles["job_title"] == custom_job) &
        (valid_profiles["experience_level"] == custom_exp)
    ]["company_size"].unique()
)
custom_size = st.selectbox("Company Size", size_options)

# ---------------------------------------------------------
# Growth-Based Helpers
# ---------------------------------------------------------
@st.cache_data
def get_mean_growth_from_similar(data, job, exp):
    similar = data[(data["job_title"] == job) & (data["experience_level"] == exp)]

    growth_rates = []

    for size in similar["company_size"].unique():
        s = data[
            (data["job_title"] == job) &
            (data["experience_level"] == exp) &
            (data["company_size"] == size)
        ].groupby("work_year")["salary_in_usd"].mean().sort_index()

        if len(s) >= 2:
            yrs = s.index.values
            vals = s.values
            g = (vals[-1] - vals[0]) / vals[0] / (yrs[-1] - yrs[0])
            growth_rates.append(float(np.clip(g, -0.20, 0.20)))

    if len(growth_rates) > 0:
        return float(np.mean(growth_rates)), len(growth_rates)

    # Fallback: job-level average growth
    j = data[data["job_title"] == job].groupby("work_year")["salary_in_usd"].mean().sort_index()
    if len(j) >= 2:
        yrs = j.index.values
        vals = j.values
        g = (vals[-1] - vals[0]) / vals[0] / (yrs[-1] - yrs[0])
        return float(np.clip(g, -0.15, 0.15)), 0

    return 0.05, 0

@st.cache_data
def calculate_growth_rate(data, job, exp, size):
    profile = data[
        (data["job_title"] == job) &
        (data["experience_level"] == exp) &
        (data["company_size"] == size)
    ].groupby("work_year")["salary_in_usd"].mean().sort_index()

    if len(profile) < 2:
        return None, None, None

    yrs = profile.index.values
    vals = profile.values

    g = (vals[-1] - vals[0]) / vals[0] / (yrs[-1] - yrs[0])
    g = float(np.clip(g, -0.20, 0.20))

    return g, yrs.tolist(), vals.tolist()

# ---------------------------------------------------------
# SALARY ENGINE — Clean & Final Version
# ---------------------------------------------------------
def get_salary(year, job, exp, size, method):
    profile_data = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ].sort_values("work_year")

    profile_years = profile_data["work_year"].tolist()
    num_years = len(profile_years)

    # ACTUAL DATA (2020–2022)
    if year in profile_years:
        actual_salary = profile_data[profile_data["work_year"] == year]["salary_in_usd"].mean()
        return actual_salary, "Actual Data"

    # CASE 1 — ≥2 years of actual data → own pattern
    if num_years >= 2:
        g, yrs, vals = calculate_growth_rate(df, job, exp, size)
        if g is not None:
            last_year = yrs[-1]
            last_salary = vals[-1]
            years_ahead = year - last_year
            prediction = last_salary * ((1 + g) ** years_ahead)
            return prediction, "Growth-Based (Own Pattern)"

    # CASE 2 — 1 actual year → similar profiles
    if num_years == 1:
        avg_growth, _ = get_mean_growth_from_similar(df, job, exp)
        base_salary = profile_data["salary_in_usd"].mean()
        base_year = profile_years[0]
        prediction = base_salary * ((1 + avg_growth) ** (year - base_year))
        return prediction, "Growth-Based (Similar Profiles)"

    # CASE 3 — No actual years (filtered out, should NEVER happen now)
    pred_input = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size]
    })
    ml_pred = selected_model.predict(pred_input)[0]
    return ml_pred, "ML Fallback (Unexpected)"
# ---------------------------------------------------------
# TAB STRUCTURE
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast (2020–2030)", "📊 Model Performance", "🛠 Feature Importance", "📁 Raw Dataset"]
)

# =========================================================
# 📈 TAB 1 — FORECAST
# =========================================================
with tab1:
    st.subheader("📈 Salary Forecast (2020–2030)")

    all_years = list(range(2020, 2031))
    forecast_data = []

    for y in all_years:
        salary, source = get_salary(
            y, custom_job, custom_exp, custom_size, prediction_method
        )
        forecast_data.append({
            "year": y,
            "salary_in_usd": salary,
            "source": source
        })

    forecast_df = pd.DataFrame(forecast_data)

    # Main Line Chart
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=forecast_df[forecast_df["source"] == "Actual Data"]["year"],
        y=forecast_df[forecast_df["source"] == "Actual Data"]["salary_in_usd"],
        mode="lines+markers",
        name="Actual Data",
        line=dict(color="#4CAF50", width=4)
    ))

    # Growth-Based
    fig.add_trace(go.Scatter(
        x=forecast_df[forecast_df["source"].str.contains("Growth")]["year"],
        y=forecast_df[forecast_df["source"].str.contains("Growth")]["salary_in_usd"],
        mode="lines+markers",
        name="Growth-Based",
        line=dict(color="#2196F3", width=3, dash="dash")
    ))

    # ML fallback (should rarely appear now)
    fig.add_trace(go.Scatter(
        x=forecast_df[forecast_df["source"].str.contains("ML")]["year"],
        y=forecast_df[forecast_df["source"].str.contains("ML")]["salary_in_usd"],
        mode="lines+markers",
        name="ML Fallback",
        line=dict(color="#FF9800", width=2, dash="dot")
    ))

    fig.update_layout(
        title="Salary Forecast for Selected Profile",
        xaxis_title="Year",
        yaxis_title="Salary (USD)",
        template="plotly_white",
        height=480
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data Table
    st.subheader("📄 Forecast Data")
    st.dataframe(forecast_df)

# =========================================================
# 📊 TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab2:
    st.subheader("📊 Machine Learning Model Performance")

    perf_df = pd.DataFrame(performance_metrics).T[
        ["R²", "MAE", "RMSE", "CV Mean", "CV Std"]
    ]

    st.dataframe(perf_df.style.highlight_max(color="lightgreen", axis=0))

    # Visual Comparison
    perf_chart = px.bar(
        perf_df.reset_index(),
        x="index",
        y="R²",
        color="index",
        title="R² Score Comparison",
        labels={"index": "Model"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(perf_chart, use_container_width=True)
# =========================================================
# 🛠 TAB 3 — FEATURE IMPORTANCE
# =========================================================
with tab3:
    st.subheader("🛠 Feature Importance")

    # Retrieve the true pipeline (not the wrong dictionary)
    pipeline = trained_models[model_type]

    # Safely detect steps
    if "prep" in pipeline.named_steps:
        prep_obj = pipeline.named_steps["prep"]
        model_obj = pipeline.named_steps["model"]
    else:
        st.warning("⚠️ No preprocessing step found in this model pipeline. Feature importance unavailable.")
        prep_obj = None
        model_obj = None

    if model_obj is not None and hasattr(model_obj, "feature_importances_"):

        ohe = prep_obj.named_transformers_["cat"]
        encoded_features = (
            list(ohe.get_feature_names_out(["job_title", "experience_level", "company_size"])) +
            ["work_year"]
        )

        fi_df = pd.DataFrame({
            "Feature": encoded_features,
            "Importance": model_obj.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(fi_df)

        fi_df["Category"] = fi_df["Feature"].apply(
            lambda x: (
                "Job Title" if x.startswith("job_title") else
                "Experience" if x.startswith("experience_level") else
                "Company Size" if x.startswith("company_size") else
                "Year"
            )
        )

        grouped = fi_df.groupby("Category")["Importance"].sum().reset_index()

        fig_imp = px.bar(
            grouped,
            x="Importance",
            y="Category",
            orientation="h",
            title="Feature Importance by Category",
            color="Category"
        )
        st.plotly_chart(fig_imp, width="stretch")

    else:
        st.info("ℹ️ Feature importance is only supported for Random Forest & Gradient Boosting (and when pipeline is intact).")


# =========================================================
# 📁 TAB 4 — RAW DATASET
# =========================================================
with tab4:
    st.subheader("📁 Raw Dataset (Filtered Valid Profiles Only)")

    st.markdown("""
    <div style='padding: 10px; background-color: #f1f8ff; border-left: 4px solid #2196f3;'>
        ✔ Only profiles with at least <b>one year of actual data</b> (2020–2022)  
        ✔ All invalid job/exp/size combinations removed  
        ✔ Ensures clean forecasting with no missing-history issues  
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; color:#666;">
    <p><b>Cybersecurity Salary Prediction Dashboard</b></p>
    <p>Powered by Streamlit · Machine Learning · Salary Intelligence Engine</p>
</div>
""", unsafe_allow_html=True)


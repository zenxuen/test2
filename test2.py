import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
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

st.markdown(
    '<h1 class="main-header">💼 Cybersecurity Salary Prediction Dashboard</h1>',
    unsafe_allow_html=True
)
st.markdown("*Actual Data: 2020-2022 | Predictions: 2023-2030*")
st.markdown("**Target:** salary_in_usd | **Predictors:** work_year, job_title, experience_level, company_size")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    # Just in case, enforce column names you told me
    # work_year, experience_level, employment_type, job_title,
    # salary, salary_currency, salary_in_usd, employee_residence,
    # remote_ratio, company_location, company_size
    return df

df = load_data()

# ---------------------------------------------------------
# Valid Profiles (remove 0-year profiles from selection)
# ---------------------------------------------------------
profile_counts = (
    df.groupby(["job_title", "experience_level", "company_size"])["work_year"]
      .nunique()
      .reset_index(name="n_years")
)

valid_profiles = profile_counts[profile_counts["n_years"] > 0].copy()

valid_jobs = sorted(valid_profiles["job_title"].unique())

# ---------------------------------------------------------
# Train Random Forest Model (pure ML engine)
# ---------------------------------------------------------
@st.cache_resource
def train_rf_model(data: pd.DataFrame):
    feature_cols = ["work_year", "job_title", "experience_level", "company_size"]
    target_col = "salary_in_usd"

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_cols = ["job_title", "experience_level", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"  # keep work_year numeric
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", rf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="r2"
    )

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "CV R2 Mean": cv_scores.mean(),
        "CV R2 Std": cv_scores.std()
    }

    return pipeline, metrics, X_test, y_test

rf_model, rf_metrics, X_test, y_test = train_rf_model(df)

# ---------------------------------------------------------
# Sidebar - Engine Selection & Info
# ---------------------------------------------------------
st.sidebar.header("🎛️ Prediction Engine")

engine = st.sidebar.radio(
    "Select Engine",
    ["Growth-Based", "Random Forest (ML)"],
    index=0,
    help=(
        "• Growth-Based: uses historical growth patterns from the dataset\n"
        "• Random Forest (ML): machine learning model trained on 2020–2022 data"
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
- **Total Records:** {len(df):,}
- **Actual Data Years:** 2020–2022
- **Prediction Years:** 2023–2030
- **Job Titles:** {df['job_title'].nunique()}
- **Avg Salary (2020–2022):** ${df['salary_in_usd'].mean():,.0f}
""")

st.sidebar.subheader("🤖 Random Forest Performance")
st.sidebar.metric("R² Score", f"{rf_metrics['R2']:.3f}")
st.sidebar.metric("MAE", f"${rf_metrics['MAE']:,.0f}")
st.sidebar.metric("RMSE", f"${rf_metrics['RMSE']:,.0f}")
st.sidebar.caption(
    f"CV R²: {rf_metrics['CV R2 Mean']:.3f} ± {rf_metrics['CV R2 Std']:.3f}"
)

# ---------------------------------------------------------
# Helper: Average Growth from Similar Profiles (for growth engine)
# ---------------------------------------------------------
@st.cache_data
def get_mean_growth_from_similar(data, job, exp):
    """
    For profiles with only 1 year of data:
    Use the average annual growth rate of similar profiles:
    - Same job + experience, across different company sizes
    - Fallback: job only
    - Fallback: experience only
    - Final fallback: default 5% growth
    """
    similar = data[
        (data["job_title"] == job) &
        (data["experience_level"] == exp)
    ]

    growth_rates = []

    if len(similar) > 0:
        for size in similar["company_size"].unique():
            prof = data[
                (data["job_title"] == job) &
                (data["experience_level"] == exp) &
                (data["company_size"] == size)
            ].groupby("work_year")["salary_in_usd"].mean().sort_index()

            if len(prof) >= 2:
                years = prof.index.values
                salaries = prof.values
                g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
                g = float(np.clip(g, -0.20, 0.20))
                growth_rates.append(g)

    if len(growth_rates) > 0:
        return float(np.mean(growth_rates)), len(growth_rates)

    job_series = data[data["job_title"] == job].groupby("work_year")["salary_in_usd"].mean().sort_index()
    if len(job_series) >= 2:
        years = job_series.index.values
        salaries = job_series.values
        g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        g = float(np.clip(g, -0.15, 0.15))
        return g, 0

    exp_series = data[data["experience_level"] == exp].groupby("work_year")["salary_in_usd"].mean().sort_index()
    if len(exp_series) >= 2:
        years = exp_series.index.values
        salaries = exp_series.values
        g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        g = float(np.clip(g, -0.15, 0.15))
        return g, 0

    return 0.05, 0

# ---------------------------------------------------------
# Helper: Own Profile Growth (for ≥2 years)
# ---------------------------------------------------------
@st.cache_data
def calculate_growth_rate(data, job, exp, size):
    profile_data = data[
        (data["job_title"] == job) &
        (data["experience_level"] == exp) &
        (data["company_size"] == size)
    ].groupby("work_year")["salary_in_usd"].mean().sort_index()

    if len(profile_data) < 2:
        return None, None, None

    years = profile_data.index.values
    salaries = profile_data.values

    first_year, last_year = years[0], years[-1]
    first_salary, last_salary = salaries[0], salaries[-1]

    if first_salary <= 0:
        return None, None, None

    growth = (last_salary - first_salary) / first_salary / (last_year - first_year)
    growth = float(np.clip(growth, -0.20, 0.20))

    return growth, years.tolist(), salaries.tolist()

# ---------------------------------------------------------
# Growth-Based Engine (NO ML inside)
# ---------------------------------------------------------
def get_salary_growth(year, job, exp, size):
    """
    Pure growth-based engine:
    - 2020–2022: if actual exists, use actual.
    - If profile has 1 year: use similar-profile growth.
    - If profile has 2+ years: use own pattern.
    - If somehow profile has 0 years (we try to avoid in selection),
      return NaN.
    """
    profile_history = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ]

    years_available = sorted(profile_history["work_year"].unique())
    num_years = len(years_available)

    # 2020–2022: if we have actual data, always use it
    if year <= 2022:
        actual_for_year = profile_history[profile_history["work_year"] == year]
        if len(actual_for_year) > 0:
            return actual_for_year["salary_in_usd"].mean(), "Actual (Growth Engine)"

    # If 0 years → we shouldn't even be here (selection filtered),
    # but just in case, return NaN
    if num_years == 0:
        return np.nan, "No Data (Growth)"

    # Exactly 1 year
    if num_years == 1:
        base_year = years_available[0]
        base_salary = profile_history["salary_in_usd"].mean()

        if year == base_year:
            return base_salary, "Actual (Single Year)"

        avg_growth, num_profiles = get_mean_growth_from_similar(df, job, exp)
        years_ahead = year - base_year
        predicted_salary = base_salary * ((1 + avg_growth) ** years_ahead)

        if num_profiles > 0:
            src = f"Predicted (Growth: {num_profiles} similar profiles, {avg_growth*100:.1f}%/yr)"
        else:
            src = f"Predicted (Growth: broader job/exp trend, {avg_growth*100:.1f}%/yr)"

        return max(0, predicted_salary), src

    # 2+ years → own pattern
    if num_years >= 2:
        growth_rate, yrs, sals = calculate_growth_rate(df, job, exp, size)

        if growth_rate is None:
            # If somehow can't compute, just return NaN
            return np.nan, "Growth Pattern Unstable"

        last_year = yrs[-1]
        last_salary = sals[-1]

        if year in yrs:
            actual_for_year = profile_history[profile_history["work_year"] == year]
            if len(actual_for_year) > 0:
                return actual_for_year["salary_in_usd"].mean(), "Actual (Multi-Year Profile)"

        years_ahead = year - last_year
        predicted_salary = last_salary * ((1 + growth_rate) ** years_ahead)

        return max(0, predicted_salary), f"Predicted (Own Growth Pattern {growth_rate*100:.1f}%/yr)"

    # Fallback
    return np.nan, "Growth Fallback"

# ---------------------------------------------------------
# Random Forest Engine (PURE ML)
# ---------------------------------------------------------
def get_salary_rf(year, job, exp, size):
    """
    Pure ML engine:
    - Uses RandomForest pipeline only.
    - For 2020–2022, we STILL show "Actual" if exists (for visual baseline),
      but the predicted value is ML-based.
    """
    profile_history = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ]

    # Actual indicator (for labeling only)
    has_actual_year = (
        (year <= 2022) and
        (len(profile_history[profile_history["work_year"] == year]) > 0)
    )

    X_input = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size]
    })

    salary_pred = rf_model.predict(X_input)[0]

    if has_actual_year:
        return max(0, salary_pred), "RF Predicted (Actual exists in dataset)"
    else:
        return max(0, salary_pred), "RF Predicted"

# ---------------------------------------------------------
# Helper to route engine
# ---------------------------------------------------------
def get_salary_engine(year, job, exp, size, engine_name):
    if engine_name == "Growth-Based":
        return get_salary_growth(year, job, exp, size)
    else:
        return get_salary_rf(year, job, exp, size)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Salary Forecast",
    "📊 Data Analysis",
    "💰 Salary Calculator",
    "🗺️ Model Insights"
])

# ---------------------------------------------------------
# TAB 1: Salary Forecast (2020–2030)
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Customize Your Profile")

    # Dependent dropdowns: only valid profiles
    col1, col2, col3 = st.columns(3)

    with col1:
        custom_job = st.selectbox("👔 Job Title", valid_jobs)

    job_filtered = valid_profiles[valid_profiles["job_title"] == custom_job]
    valid_exps = sorted(job_filtered["experience_level"].unique())

    with col2:
        custom_exp = st.selectbox("📈 Experience Level", valid_exps)

    exp_filtered = job_filtered[job_filtered["experience_level"] == custom_exp]
    valid_sizes = sorted(exp_filtered["company_size"].unique())

    with col3:
        custom_size = st.selectbox("🏢 Company Size", valid_sizes)

    profile_history = df[
        (df["job_title"] == custom_job) &
        (df["experience_level"] == custom_exp) &
        (df["company_size"] == custom_size)
    ]
    unique_years = profile_history["work_year"].nunique()

    # Info messages for growth engine
    if engine == "Growth-Based":
        if unique_years == 1:
            year_available = profile_history["work_year"].iloc[0]
            salary_available = profile_history["salary_in_usd"].mean()
            avg_growth, num_profiles = get_mean_growth_from_similar(df, custom_job, custom_exp)

            if num_profiles > 0:
                st.info(
                    f"ℹ️ Growth Engine: This profile has **1 year** of data "
                    f"({year_available}: ${salary_available:,.0f}). Using "
                    f"**average pattern** from {num_profiles} similar profiles "
                    f"≈ **{avg_growth*100:.1f}%/year**."
                )
            else:
                st.info(
                    f"ℹ️ Growth Engine: Only **1 year** of data "
                    f"({year_available}: ${salary_available:,.0f}). Using "
                    f"broader job/experience trend ≈ {avg_growth*100:.1f}%/year."
                )
        elif unique_years >= 2:
            growth_rate, years_list, salaries_list = calculate_growth_rate(
                df, custom_job, custom_exp, custom_size
            )
            if growth_rate is not None:
                salary_progression = " → ".join(
                    [f"${s:,.0f}" for s in salaries_list]
                )
                if growth_rate < 0:
                    st.warning(
                        f"📉 Growth Engine: **{unique_years} years** of data "
                        f"({salary_progression}), pattern ≈ "
                        f"{growth_rate*100:.1f}%/year (declining)."
                    )
                else:
                    st.success(
                        f"✅ Growth Engine: **{unique_years} years** of data "
                        f"({salary_progression}), pattern ≈ "
                        f"{growth_rate*100:.1f}%/year."
                    )
            else:
                st.info(
                    "ℹ️ Growth Engine: Multiple years exist, but growth "
                    "pattern is unstable; forecasts may be conservative."
                )
        else:
            st.warning("⚠️ Growth Engine: No usable history (should not happen, selection is filtered).")

    else:
        st.info(
            "🤖 Random Forest Engine: Predictions are fully ML-based. "
            "2020–2022 points are still shown where data exists."
        )

    st.markdown("---")

    # Build forecast 2020–2030 with selected engine
    all_years = np.arange(2020, 2031)
    forecast_records = []
    for y in all_years:
        salary, src = get_salary_engine(y, custom_job, custom_exp, custom_size, engine)
        forecast_records.append({
            "Year": y,
            "Salary (USD)": salary,
            "Source": src
        })

    forecast_df = pd.DataFrame(forecast_records)
    safe_df = forecast_df.dropna(subset=["Salary (USD)"])

    # Split "Actual" vs "Predicted" by label
    actual_mask = forecast_df["Source"].str.contains("Actual", na=False)
    actual_df = forecast_df[actual_mask]
    predicted_df = forecast_df[~actual_mask]

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        if len(actual_df) > 0:
            st.success(
                f"✅ **Actual-like Points ({len(actual_df)}):** "
                f"{', '.join(map(str, actual_df['Year'].tolist()))}"
            )
        else:
            st.warning("⚠️ No actual-year labels detected for this profile.")
    with col_info2:
        if len(predicted_df) > 0:
            st.info(
                f"🔮 **Predicted Points ({len(predicted_df)}):** "
                f"{', '.join(map(str, predicted_df['Year'].tolist()))}"
            )
        else:
            st.warning("⚠️ No predicted points (check data).")

    # Summary metrics from safe_df
    if not safe_df.empty:
        if 2020 in safe_df["Year"].values:
            start_salary = float(
                safe_df[safe_df["Year"] == 2020]["Salary (USD)"].iloc[0]
            )
        else:
            start_salary = float(safe_df.iloc[0]["Salary (USD)"])

        if 2030 in safe_df["Year"].values:
            end_salary = float(
                safe_df[safe_df["Year"] == 2030]["Salary (USD)"].iloc[0]
            )
        else:
            end_salary = float(safe_df.iloc[-1]["Salary (USD)"])

        if 2025 in safe_df["Year"].values:
            mid_salary = float(
                safe_df[safe_df["Year"] == 2025]["Salary (USD)"].iloc[0]
            )
        else:
            mid_salary = end_salary

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📅 2020 Salary", f"${start_salary:,.0f}")
        with col_b:
            st.metric("📅 2025 Salary", f"${mid_salary:,.0f}")
        with col_c:
            if start_salary > 0:
                total_growth = ((end_salary - start_salary) / start_salary) * 100
                st.metric("📈 Growth 2020–2030", f"{total_growth:.1f}%")
            else:
                st.metric("📈 Growth 2020–2030", "N/A")

    # Plot
    fig = go.Figure()

    if len(actual_df) > 0:
        fig.add_trace(go.Scatter(
            x=actual_df["Year"],
            y=actual_df["Salary (USD)"],
            mode="lines+markers",
            name="Actual / Historical Points",
            line=dict(color="#10b981", width=4),
            marker=dict(size=10, color="#10b981", line=dict(width=2, color="white")),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.15)",
            hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> %{text}<extra></extra>",
            text=actual_df["Source"]
        ))

    if len(predicted_df) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_df["Year"],
            y=predicted_df["Salary (USD)"],
            mode="lines+markers",
            name=f"Predictions ({engine})",
            line=dict(color="#667eea", width=4, dash="dash"),
            marker=dict(size=10, color="#764ba2", line=dict(width=2, color="white")),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.15)",
            hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> %{text}<extra></extra>",
            text=predicted_df["Source"]
        ))

    fig.update_layout(
        title=dict(
            text=f"Salary Forecast ({engine})<br><sub>{custom_job} | {custom_exp} | {custom_size}</sub>",
            font=dict(size=20)
        ),
        xaxis_title="Year",
        yaxis_title="Salary (USD)",
        template="plotly_white",
        hovermode="x unified",
        height=550,
        xaxis=dict(dtick=1),
        showlegend=True,
        font=dict(size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Detailed Forecast Table"):
        table_df = forecast_df.copy()
        table_df["Salary (USD)"] = table_df["Salary (USD)"].apply(
            lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
        )
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )

# ---------------------------------------------------------
# TAB 2: Data Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution (Actual 2020–2022)")

    col1, col2 = st.columns(2)

    with col1:
        exp_avg = (
            df.groupby("experience_level")["salary_in_usd"]
              .mean()
              .sort_values(ascending=False)
              .reset_index()
        )
        fig_exp = px.bar(
            exp_avg,
            x="experience_level",
            y="salary_in_usd",
            title="Average Salary by Experience Level",
            color="salary_in_usd",
            color_continuous_scale="Viridis",
            text="salary_in_usd"
        )
        fig_exp.update_traces(texttemplate='$%{text:,.0f}', textposition="outside")
        fig_exp.update_layout(
            showlegend=False,
            xaxis_title="Experience Level",
            yaxis_title="Avg Salary (USD)"
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    with col2:
        size_avg = (
            df.groupby("company_size")["salary_in_usd"]
              .mean()
              .sort_values(ascending=False)
              .reset_index()
        )
        fig_size = px.bar(
            size_avg,
            x="company_size",
            y="salary_in_usd",
            title="Average Salary by Company Size",
            color="salary_in_usd",
            color_continuous_scale="Plasma",
            text="salary_in_usd"
        )
        fig_size.update_traces(texttemplate='$%{text:,.0f}', textposition="outside")
        fig_size.update_layout(
            showlegend=False,
            xaxis_title="Company Size",
            yaxis_title="Avg Salary (USD)"
        )
        st.plotly_chart(fig_size, use_container_width=True)

    st.markdown("---")
    st.subheader("💎 Top 10 Highest Paying Jobs (per Year)")

    year_selector = st.radio(
        "Select Year",
        [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030],
        horizontal=True,
        index=2
    )

    if year_selector <= 2022:
        # Use actual data
        top_jobs = (
            df[df["work_year"] == year_selector]
            .groupby("job_title")["salary_in_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        data_type = "Actual Data"
    else:
        # Use selected engine for prediction (Growth vs RF)
        all_jobs = df["job_title"].unique()
        job_salaries = []

        for job in all_jobs:
            job_data = df[df["job_title"] == job]
            most_common_exp = job_data["experience_level"].mode()[0]
            most_common_size = job_data["company_size"].mode()[0]

            salary, _ = get_salary_engine(
                year_selector,
                job,
                most_common_exp,
                most_common_size,
                engine
            )
            job_salaries.append({
                "job_title": job,
                "salary_in_usd": salary
            })

        top_jobs = (
            pd.DataFrame(job_salaries)
            .sort_values("salary_in_usd", ascending=False)
            .head(10)
        )
        data_type = f"Predictions ({engine})"

    if len(top_jobs) > 0:
        fig_top = px.bar(
            top_jobs,
            x="salary_in_usd",
            y="job_title",
            orientation="h",
            title=f"Top 10 Jobs in {year_selector} — {data_type}",
            color="salary_in_usd",
            color_continuous_scale="Turbo",
            text="salary_in_usd"
        )
        fig_top.update_traces(texttemplate='$%{text:,.0f}', textposition="outside")
        fig_top.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
            height=500,
            xaxis_title="Avg Salary (USD)",
            yaxis_title=""
        )
        st.plotly_chart(fig_top, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Salary Calculator
# ---------------------------------------------------------
with tab3:
    st.subheader("💰 Salary Calculator")

    calc_year = st.slider("📅 Select Year", 2020, 2030, 2025)

    salary_value, data_source = get_salary_engine(
        calc_year, custom_job, custom_exp, custom_size, engine
    )

    st.markdown("---")

    if "Actual" in data_source:
        st.success(f"✅ {data_source}")
    elif "RF Predicted" in data_source:
        st.info(f"🤖 {data_source}")
    else:
        st.info(f"🔮 {data_source}")

    col_x, col_y, col_z = st.columns([1, 2, 1])
    with col_y:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Salary for {calc_year}</h3>
            <h1 style="font-size: 3.5rem; margin: 1rem 0;">${salary_value:,.0f}</h1>
            <p style="font-size: 1.1rem;">{custom_job}</p>
            <p>{custom_exp} | {custom_size}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Market Comparison (Historical 2020–2022)")

    market_avg_all = df[df["job_title"] == custom_job]["salary_in_usd"].mean()

    if calc_year <= 2022:
        market_year = df[
            (df["job_title"] == custom_job) &
            (df["work_year"] == calc_year)
        ]["salary_in_usd"].mean()
        if not np.isnan(market_year):
            market_avg = market_year
            market_label = f"Market Avg ({calc_year})"
        else:
            market_avg = market_avg_all
            market_label = "Overall Market Avg"
    else:
        market_avg = market_avg_all
        market_label = "Historical Market Avg (2020–2022)"

    diff = salary_value - market_avg
    diff_pct = (diff / market_avg) * 100 if market_avg > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Salary", f"${salary_value:,.0f}")
    with col2:
        st.metric(market_label, f"${market_avg:,.0f}")
    with col3:
        st.metric("Difference", f"${diff:,.0f}", f"{diff_pct:+.1f}%")

# ---------------------------------------------------------
# TAB 4: Model Insights
# ---------------------------------------------------------
with tab4:
    st.subheader("🗺️ Model Performance & Insights")

    st.markdown("""
    <div class="info-box">
        <h4>📌 Engines Overview</h4>
        <ul>
            <li><b>Growth-Based Engine</b>: Uses historical patterns (year-by-year growth) from the dataset. No machine learning inside.</li>
            <li><b>Random Forest (ML)</b>: Trained on 2020–2022 salaries using
                <code>work_year</code>, <code>job_title</code>, <code>experience_level</code>, <code>company_size</code>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Random Forest Metrics")
    metric_df = pd.DataFrame([{
        "Model": "Random Forest",
        "R²": rf_metrics["R2"],
        "MAE": rf_metrics["MAE"],
        "RMSE": rf_metrics["RMSE"]
    }])
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Feature Importance (Random Forest)")

    rf_estimator = rf_model.named_steps["model"]
    preprocessor = rf_model.named_steps["prep"]
    ohe = preprocessor.named_transformers_["cat"]

    importance_df = pd.DataFrame({
        "Importance": rf_estimator.feature_importances_
    })

    n_job = len(ohe.categories_[0])
    n_exp = len(ohe.categories_[1])
    n_size = len(ohe.categories_[2])

    feature_types = (
        ["Job Title"] * n_job +
        ["Experience Level"] * n_exp +
        ["Company Size"] * n_size +
        ["Year"]  # passthrough work_year
    )

    importance_df["Feature Type"] = feature_types

    importance_grouped = (
        importance_df
        .groupby("Feature Type")["Importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_imp = px.bar(
        importance_grouped,
        x="Importance",
        y="Feature Type",
        orientation="h",
        title="Feature Importance (Grouped by Type)",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><b>Built with Streamlit • Growth Engine + Random Forest (ML)</b></p>
        <p style='font-size: 0.9rem;'>Active Engine: {engine} | Random Forest R²: {rf_metrics['R2']:.3f} | MAE: ${rf_metrics['MAE']:,.0f}</p>
        <p style='font-size: 0.8rem;'>Target: salary_in_usd | Predictors: work_year, job_title, experience_level, company_size</p>
    </div>
""", unsafe_allow_html=True)

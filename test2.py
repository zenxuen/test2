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
st.markdown("*Actual Data: 2020-2022 | ML Predictions: 2023-2030*")
st.markdown("**Target:** Salary | **Predictors:** Year, Job Title, Experience, Company Size")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar - Model Selection & Info
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
    help="Choose the prediction algorithm"
)

# Prediction method selector (for explanations / future use)
prediction_method = st.sidebar.radio(
    "Prediction Method",
    ["Growth-Based (Recommended)", "Pure ML Model", "Hybrid"],
    help="Growth-Based: Uses historical growth patterns\nPure ML: Uses model only\nHybrid: Blends both approaches"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
- **Total Records:** {len(df):,}
- **Actual Data Years:** 2020-2022
- **Prediction Years:** 2023-2030
- **Job Titles:** {df['job_title'].nunique()}
- **Avg Salary (2020-2022):** ${df['salary_in_usd'].mean():,.0f}
""")

# ---------------------------------------------------------
# Train Models - Target: salary_in_usd, Predictors: All others
# ---------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    """
    Train three models (Linear, RF, GB) on:
    - Target: salary_in_usd
    - Features: work_year, job_title, experience_level, company_size
    """
    X = data[["work_year", "job_title", "experience_level", "company_size"]]
    y = data["salary_in_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = ["job_title", "experience_level", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"  # Keep work_year as numeric
    )

    models_dict = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=7,
            learning_rate=0.1
        )
    }

    trained_models = {}
    metrics = {}

    for name, estimator in models_dict.items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring='r2'
        )

        trained_models[name] = pipeline
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV R² Mean": cv_scores.mean(),
            "CV R² Std": cv_scores.std()
        }

    return trained_models, metrics, X_test, y_test


models, performance_metrics, X_test, y_test = train_models(df)
selected_model = models[model_type]

st.sidebar.success(f"**Active Model:** {model_type}")

# ---------------------------------------------------------
# Display Model Performance
# ---------------------------------------------------------
with st.sidebar.expander("📈 Model Performance", expanded=True):
    perf = performance_metrics[model_type]
    st.metric("R² Score", f"{perf['R² Score']:.3f}")
    st.metric("MAE", f"${perf['MAE']:,.0f}")
    st.metric("RMSE", f"${perf['RMSE']:,.0f}")
    st.caption(f"CV: {perf['CV R² Mean']:.3f}±{perf['CV R² Std']:.3f}")

# ---------------------------------------------------------
# Helper: Average Growth from Similar Profiles
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

    # Same job + experience, split by size
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

    # Fallback: job-only
    job_series = data[data["job_title"] == job].groupby("work_year")["salary_in_usd"].mean().sort_index()
    if len(job_series) >= 2:
        years = job_series.index.values
        salaries = job_series.values
        g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        g = float(np.clip(g, -0.15, 0.15))
        return g, 0

    # Fallback: experience-only
    exp_series = data[data["experience_level"] == exp].groupby("work_year")["salary_in_usd"].mean().sort_index()
    if len(exp_series) >= 2:
        years = exp_series.index.values
        salaries = exp_series.values
        g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        g = float(np.clip(g, -0.15, 0.15))
        return g, 0

    # Final fallback
    return 0.05, 0

# ---------------------------------------------------------
# Helper: Own Profile Growth (for ≥2 years)
# ---------------------------------------------------------
@st.cache_data
def calculate_growth_rate(data, job, exp, size):
    """
    Calculate annual growth rate using this specific profile's own history.
    Requires at least 2 years of data for (job, exp, size).
    """
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
# Prediction Function (Growth-based logic)
# ---------------------------------------------------------
def get_salary(year, job, exp, size, method="Growth-Based (Recommended)"):
    """
    Main prediction function:
    1) If exact actual exists (2020–2022) → return actual.
    2) If profile has 0 years of data → ML-only (RandomForest / selected model).
    3) If profile has 1 year → use similar-profile growth.
    4) If profile has ≥2 years → use own profile pattern.
    """
    profile_history = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ]

    years_available = sorted(profile_history["work_year"].unique())
    num_years = len(years_available)

    # 1) For 2020-2022: use actual where available
    if year <= 2022:
        actual_for_year = profile_history[profile_history["work_year"] == year]
        if len(actual_for_year) > 0:
            return actual_for_year["salary_in_usd"].mean(), "Actual"

    # 0-year history → ML only
    if num_years == 0:
        pred_input = pd.DataFrame({
            "work_year": [year],
            "job_title": [job],
            "experience_level": [exp],
            "company_size": [size]
        })
        predicted_salary = selected_model.predict(pred_input)[0]
        return max(0, predicted_salary), "Predicted (ML Only - No Profile History)"

    # 1-year history → similar-profile pattern
    if num_years == 1:
        base_year = years_available[0]
        base_salary = profile_history["salary_in_usd"].mean()

        if year == base_year:
            return base_salary, "Actual (Single Year)"

        avg_growth, num_profiles = get_mean_growth_from_similar(df, job, exp)
        years_ahead = year - base_year
        predicted_salary = base_salary * ((1 + avg_growth) ** years_ahead)

        if num_profiles > 0:
            src = f"Predicted (Similar Profiles Pattern, {num_profiles} profiles)"
        else:
            src = "Predicted (Job/Experience-Level Pattern)"

        return max(0, predicted_salary), src

    # ≥ 2-year history → own profile pattern
    if num_years >= 2:
        growth_rate, yrs, sals = calculate_growth_rate(df, job, exp, size)

        if growth_rate is None:
            pred_input = pd.DataFrame({
                "work_year": [year],
                "job_title": [job],
                "experience_level": [exp],
                "company_size": [size]
            })
            predicted_salary = selected_model.predict(pred_input)[0]
            return max(0, predicted_salary), "Predicted (ML Fallback)"

        last_year = yrs[-1]
        last_salary = sals[-1]

        if year in yrs:
            actual_for_year = profile_history[profile_history["work_year"] == year]
            if len(actual_for_year) > 0:
                return actual_for_year["salary_in_usd"].mean(), "Actual"

        years_ahead = year - last_year
        predicted_salary = last_salary * ((1 + growth_rate) ** years_ahead)

        return max(0, predicted_salary), "Predicted (Own Profile Pattern)"

    # Safety fallback
    pred_input = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size]
    })
    predicted_salary = selected_model.predict(pred_input)[0]
    return max(0, predicted_salary), "Predicted (Fallback)"

# ---------------------------------------------------------
# Main Content - Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Salary Forecast", 
    "📊 Data Analysis", 
    "💰 Salary Calculator", 
    "🗺️ Model Insights",
    "🏆 Job Rankings (2023-2030)"
])

# ---------------------------------------------------------
# TAB 1: Salary Forecast (2020-2030)
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Customize Your Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        custom_job = st.selectbox("👔 Job Title", sorted(df["job_title"].unique()))
    with col2:
        custom_exp = st.selectbox("📈 Experience Level", sorted(df["experience_level"].unique()))
    with col3:
        custom_size = st.selectbox("🏢 Company Size", sorted(df["company_size"].unique()))

    profile_history = df[
        (df["job_title"] == custom_job) &
        (df["experience_level"] == custom_exp) &
        (df["company_size"] == custom_size)
    ]
    unique_years = profile_history["work_year"].nunique()

    if unique_years == 0:
        st.warning(
            "⚠️ No actual historical data for this exact profile "
            f"({custom_job}, {custom_exp}, {custom_size}). "
            "Predictions will rely more on the ML model and broader patterns."
        )
    elif unique_years == 1:
        year_available = profile_history['work_year'].iloc[0]
        salary_available = profile_history['salary_in_usd'].mean()
        avg_growth, num_profiles = get_mean_growth_from_similar(df, custom_job, custom_exp)

        if num_profiles > 0:
            if avg_growth < 0:
                st.warning(
                    f"📉 Only **1 year** of data ({year_available}: "
                    f"${salary_available:,.0f}). Using **average pattern from "
                    f"{num_profiles} similar profiles**: "
                    f"**{avg_growth*100:.1f}% per year (declining)**."
                )
            else:
                st.info(
                    f"ℹ️ Only **1 year** of data ({year_available}: "
                    f"${salary_available:,.0f}). Using **average pattern from "
                    f"{num_profiles} similar profiles**: "
                    f"**{avg_growth*100:.1f}% per year**."
                )
        else:
            if avg_growth < 0:
                st.warning(
                    f"📉 Only **1 year** of data ({year_available}: "
                    f"${salary_available:,.0f}). Using broader "
                    f"job/experience-level trend: "
                    f"**{avg_growth*100:.1f}% per year (declining)**."
                )
            else:
                st.info(
                    f"ℹ️ Only **1 year** of data ({year_available}: "
                    f"${salary_available:,.0f}). Using broader "
                    f"job/experience-level trend: "
                    f"**{avg_growth*100:.1f}% per year**."
                )
    else:
        growth_rate, years_list, salaries_list = calculate_growth_rate(
            df, custom_job, custom_exp, custom_size
        )
        if growth_rate is not None:
            salary_progression = " → ".join(
                [f"${s:,.0f}" for s in salaries_list]
            )
            if growth_rate < 0:
                st.warning(
                    f"📉 This profile has **{unique_years} years** of data: "
                    f"{salary_progression}. **Declining pattern**: "
                    f"**{growth_rate*100:.1f}% per year** (own profile pattern)."
                )
            else:
                st.success(
                    f"✅ This profile has **{unique_years} years** of data: "
                    f"{salary_progression}. **Growth pattern**: "
                    f"**{growth_rate*100:.1f}% per year** (own profile pattern)."
                )
        else:
            st.info(
                "ℹ️ Multiple years of data exist, but growth pattern could not be "
                "reliably estimated. ML model will assist in predictions."
            )

    st.markdown("---")

    # Generate forecast for 2020-2030
    all_years = np.arange(2020, 2031)
    forecast_data = []

    for year in all_years:
        salary, source = get_salary(
            year, custom_job, custom_exp, custom_size, prediction_method
        )
        forecast_data.append({
            "Year": year,
            "Salary (USD)": salary,
            "Source": source
        })

    forecast_df = pd.DataFrame(forecast_data)

    if (forecast_df["Salary (USD)"] < 0).any():
        st.error(
            "⚠️ Error: Negative salaries detected! This profile may have "
            "insufficient or unstable data."
        )

    actual_mask = forecast_df["Source"].str.contains("Actual", na=False)
    actual_df = forecast_df[actual_mask]
    predicted_df = forecast_df[~actual_mask]

    actual_years = sorted(actual_df["Year"].tolist())
    predicted_years = sorted(predicted_df["Year"].tolist())

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        if actual_years:
            st.success(
                f"✅ **Actual Data ({len(actual_years)} years):** "
                f"{', '.join(map(str, actual_years))}"
            )
        else:
            st.warning("⚠️ No exact actual data for this profile in 2020-2022.")
    with col_info2:
        if predicted_years:
            preview = ", ".join(map(str, predicted_years[:5]))
            more = ", ..." if len(predicted_years) > 5 else ""
            st.info(
                f"🔮 **Predicted Data ({len(predicted_years)} years):** "
                f"{preview}{more}"
            )
        else:
            st.warning("⚠️ No predictions generated.")

    safe_df = forecast_df.dropna(subset=["Salary (USD)"])
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
            salary_2025 = float(
                safe_df[safe_df["Year"] == 2025]["Salary (USD)"].iloc[0]
            )
        else:
            salary_2025 = end_salary

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📅 2020 Salary", f"${start_salary:,.0f}")
        with col_b:
            st.metric("📅 2025 Salary (Predicted)", f"${salary_2025:,.0f}")
        with col_c:
            if start_salary > 0:
                growth = ((end_salary - start_salary) / start_salary) * 100
                st.metric("📈 Total Growth (2020-2030)", f"{growth:.1f}%")
            else:
                st.metric("📈 Total Growth (2020-2030)", "N/A")

    fig = go.Figure()

    if len(actual_df) > 0:
        fig.add_trace(go.Scatter(
            x=actual_df["Year"],
            y=actual_df["Salary (USD)"],
            mode='lines+markers',
            name='Actual Data (2020-2022)',
            line=dict(color='#10b981', width=5),
            marker=dict(size=14, color='#10b981', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.15)',
            hovertemplate='<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> Actual<extra></extra>'
        ))

    if len(predicted_df) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_df["Year"],
            y=predicted_df["Salary (USD)"],
            mode='lines+markers',
            name='Predictions (Pattern / ML)',
            line=dict(color='#667eea', width=5, dash='dash'),
            marker=dict(size=14, color='#764ba2', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.15)',
            hovertemplate='<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> %{text}<extra></extra>',
            text=predicted_df["Source"]
        ))
    else:
        st.warning("⚠️ No predicted data to display. Check if there's historical data for this profile.")

    if len(actual_df) > 0 and len(predicted_df) > 0:
        fig.add_vline(
            x=2022.5,
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text="Actual | Predictions",
            annotation_position="top"
        )

    fig.update_layout(
        title=dict(
            text=f"Salary Forecast: {custom_job}<br><sub>{custom_exp} | {custom_size}</sub>",
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
        display_df = forecast_df.copy()
        display_df['Salary (USD)'] = display_df['Salary (USD)'].apply(
            lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
        )

        st.caption("""
        **Data Sources:**
        - **Actual / Actual (Single Year):** Real data from dataset
        - **Predicted (Own Profile Pattern):** 2+ years of data for this profile, using its own growth rate
        - **Predicted (Similar Profiles Pattern):** Only 1 year of data; using average growth from similar job+experience profiles
        - **Predicted (Job/Experience-Level Pattern):** Using broader job/experience salary trend
        - **Predicted (ML Only - No Profile History):** No direct history for this profile; ML model estimate
        """)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Year": st.column_config.NumberColumn("Year", format="%d"),
                "Salary (USD)": st.column_config.TextColumn("Salary"),
                "Source": st.column_config.TextColumn("Data Source")
            }
        )

# ---------------------------------------------------------
# TAB 2: Data Analysis (no more job forecast here)
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution Analysis (Actual Data: 2020-2022)")

    col1, col2 = st.columns(2)

    with col1:
        exp_avg = df.groupby("experience_level")["salary_in_usd"].mean().sort_values(ascending=False).reset_index()
        fig_exp = px.bar(
            exp_avg,
            x="experience_level",
            y="salary_in_usd",
            title="Average Salary by Experience Level",
            color="salary_in_usd",
            color_continuous_scale="Viridis",
            text="salary_in_usd"
        )
        fig_exp.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_exp.update_layout(
            showlegend=False,
            xaxis_title="Experience Level",
            yaxis_title="Avg Salary (USD)"
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    with col2:
        size_avg = df.groupby("company_size")["salary_in_usd"].mean().sort_values(ascending=False).reset_index()
        fig_size = px.bar(
            size_avg,
            x="company_size",
            y="salary_in_usd",
            title="Average Salary by Company Size",
            color="salary_in_usd",
            color_continuous_scale="Plasma",
            text="salary_in_usd"
        )
        fig_size.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_size.update_layout(
            showlegend=False,
            xaxis_title="Company Size",
            yaxis_title="Avg Salary (USD)"
        )
        st.plotly_chart(fig_size, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Salary Calculator
# ---------------------------------------------------------
with tab3:
    st.subheader("💰 Salary Calculator")

    calc_year = st.slider("📅 Select Year", 2020, 2030, 2025)

    salary_value, data_source = get_salary(
        calc_year, custom_job, custom_exp, custom_size, prediction_method
    )

    st.markdown("---")

    if "Actual" in data_source:
        st.success(f"✅ Using {data_source} from dataset")
    elif "ML Only" in data_source:
        st.warning(f"⚠️ {data_source} (no direct profile history)")
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
    st.subheader("📊 Market Comparison")

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
        market_label = "Historical Market Avg (2020-2022)"

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
        <h4>📌 How This Works:</h4>
        <ul>
            <li><b>Target Variable:</b> salary_in_usd (what we predict)</li>
            <li><b>Predictor Variables:</b> work_year, job_title, experience_level, company_size</li>
            <li><b>Training Data:</b> Actual salaries from 2020-2022</li>
            <li><b>Predictions:</b> 2023-2030 forecasts using:
                <ul>
                    <li>Own profile pattern (if 2+ years of data)</li>
                    <li>Similar profiles pattern (if 1 year of data)</li>
                    <li>ML-only estimates (if no profile history)</li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Comparison")

    comparison_data = []
    for model_name, metrics in performance_metrics.items():
        comparison_data.append({
            "Model": model_name,
            "R² Score": metrics["R² Score"],
            "MAE ($)": metrics["MAE"],
            "RMSE ($)": metrics["RMSE"]
        })

    comparison_df = pd.DataFrame(comparison_data)

    col1, col2 = st.columns(2)

    with col1:
        fig_r2 = px.bar(
            comparison_df,
            x="Model",
            y="R² Score",
            title="R² Score (Higher = Better Fit)",
            color="R² Score",
            color_continuous_scale="Greens",
            text="R² Score"
        )
        fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_mae = px.bar(
            comparison_df,
            x="Model",
            y="MAE ($)",
            title="Mean Absolute Error (Lower = Better)",
            color="MAE ($)",
            color_continuous_scale="Reds_r",
            text="MAE ($)"
        )
        fig_mae.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("### 📈 Detailed Metrics")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if model_type in ["Random Forest", "Gradient Boosting"]:
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance")

        model_estimator = selected_model.named_steps['model']
        if hasattr(model_estimator, 'feature_importances_'):
            preprocessor = selected_model.named_steps['prep']
            ohe = preprocessor.named_transformers_['cat']

            importance_df = pd.DataFrame({
                'Importance': model_estimator.feature_importances_
            })

            n_job = len(ohe.categories_[0])
            n_exp = len(ohe.categories_[1])
            n_size = len(ohe.categories_[2])

            feature_types = (
                ['Job Title'] * n_job +
                ['Experience'] * n_exp +
                ['Company Size'] * n_size +
                ['Year']  # work_year passthrough
            )
            importance_df['Feature Type'] = feature_types

            importance_grouped = importance_df.groupby(
                'Feature Type'
            )['Importance'].sum().sort_values(
                ascending=False
            ).reset_index()

            fig_imp = px.bar(
                importance_grouped,
                x='Importance',
                y='Feature Type',
                orientation='h',
                title=f"Feature Importance - {model_type}",
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

            st.caption("💡 This shows which factors have the most impact on salary predictions")

# ---------------------------------------------------------
# TAB 5: Job Rankings 2023-2030 (Growth vs Random Forest)
# ---------------------------------------------------------
with tab5:
    st.subheader("🏆 Top 10 Jobs (2023–2030)")
    st.caption(
        "Only job titles with at least **2 different years** of historical data "
        "(2020–2022) are included in this ranking."
    )

    rank_year = st.slider("📅 Select Forecast Year", 2023, 2030, 2030)

    # Filter job titles with >= 2 years of history
    job_year_counts = df.groupby("job_title")["work_year"].nunique()
    valid_jobs = job_year_counts[job_year_counts >= 2].index.tolist()

    if not valid_jobs:
        st.warning("No job titles have 2 or more historical years in this dataset.")
    else:
        if "Random Forest" not in models:
            st.error("Random Forest model is not available. Please check the training configuration.")
        else:
            rf_pipeline = models["Random Forest"]

            ranking_rows = []
            for job in valid_jobs:
                job_data = df[df["job_title"] == job]

                exp_mode = job_data["experience_level"].mode()[0]
                size_mode = job_data["company_size"].mode()[0]

                # Growth-based salary (using your growth logic)
                growth_salary, _ = get_salary(rank_year, job, exp_mode, size_mode)

                # Random Forest salary (pure ML)
                pred_input = pd.DataFrame({
                    "work_year": [rank_year],
                    "job_title": [job],
                    "experience_level": [exp_mode],
                    "company_size": [size_mode]
                })
                rf_salary = float(rf_pipeline.predict(pred_input)[0])

                ranking_rows.append({
                    "job_title": job,
                    "experience_level": exp_mode,
                    "company_size": size_mode,
                    "Growth-Based Salary (USD)": growth_salary,
                    "Random Forest Salary (USD)": rf_salary
                })

            ranking_df = pd.DataFrame(ranking_rows)

            top_growth = ranking_df.sort_values(
                "Growth-Based Salary (USD)", ascending=False
            ).head(10)
            top_rf = ranking_df.sort_values(
                "Random Forest Salary (USD)", ascending=False
            ).head(10)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(f"### 🔮 Growth-Based Top 10 ({rank_year})")
                fig_g = px.bar(
                    top_growth,
                    x="Growth-Based Salary (USD)",
                    y="job_title",
                    orientation="h",
                    title=f"Growth-Based Top 10 Jobs in {rank_year}",
                    text="Growth-Based Salary (USD)",
                    color="Growth-Based Salary (USD)",
                    color_continuous_scale="Blues"
                )
                fig_g.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig_g.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Salary (USD)",
                    yaxis_title="Job Title",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_g, use_container_width=True)

                with st.expander("📋 Growth-Based Ranking Table"):
                    show_growth = top_growth.copy()
                    show_growth["Growth-Based Salary (USD)"] = show_growth["Growth-Based Salary (USD)"].map(
                        lambda v: f"${v:,.0f}"
                    )
                    st.dataframe(
                        show_growth[
                            ["job_title", "experience_level", "company_size", "Growth-Based Salary (USD)"]
                        ],
                        hide_index=True,
                        use_container_width=True
                    )

            with col_right:
                st.markdown(f"### 🌲 Random Forest Top 10 ({rank_year})")
                fig_rf = px.bar(
                    top_rf,
                    x="Random Forest Salary (USD)",
                    y="job_title",
                    orientation="h",
                    title=f"Random Forest Top 10 Jobs in {rank_year}",
                    text="Random Forest Salary (USD)",
                    color="Random Forest Salary (USD)",
                    color_continuous_scale="Greens"
                )
                fig_rf.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig_rf.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Salary (USD)",
                    yaxis_title="Job Title",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig_rf, use_container_width=True)

                with st.expander("📋 Random Forest Ranking Table"):
                    show_rf = top_rf.copy()
                    show_rf["Random Forest Salary (USD)"] = show_rf["Random Forest Salary (USD)"].map(
                        lambda v: f"${v:,.0f}"
                    )
                    st.dataframe(
                        show_rf[
                            ["job_title", "experience_level", "company_size", "Random Forest Salary (USD)"]
                        ],
                        hide_index=True,
                        use_container_width=True
                    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><b>Built with Streamlit • Powered by Machine Learning</b></p>
        <p style='font-size: 0.9rem;'>Active Model: {model_type} | R² Score: {performance_metrics[model_type]["R² Score"]:.3f} | MAE: ${performance_metrics[model_type]["MAE"]:,.0f}</p>
        <p style='font-size: 0.8rem;'>Target: salary_in_usd | Predictors: work_year, job_title, experience_level, company_size</p>
    </div>
""", unsafe_allow_html=True)

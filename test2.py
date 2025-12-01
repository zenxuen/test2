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
st.markdown("**Target:** salary_in_usd &nbsp;&nbsp;|&nbsp;&nbsp; **Features:** work_year, job_title, experience_level, company_size")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Expecting columns like:
    # work_year, experience_level, employment_type, job_title,
    # salary, salary_currency, salary_in_usd, employee_residence,
    # remote_ratio, company_location, company_size
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

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
- **Total Records:** {len(df):,}
- **Actual Data Years:** 2020–2022
- **Prediction Years (ML):** 2023–2030
- **Job Titles:** {df['job_title'].nunique()}
- **Avg Salary (2020–2022):** ${df['salary_in_usd'].mean():,.0f}
""")

# ---------------------------------------------------------
# Train Models - Target: salary_in_usd, Features: work_year, job_title, experience_level, company_size
# ---------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    # FEATURES (X) and TARGET (y)
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
        remainder="passthrough"  # keep work_year as numeric
    )

    models = {
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
            max_depth=3,          # GB max_depth must be small
            learning_rate=0.1
        )
    }

    pipelines = {}
    metrics = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", estimator)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="r2"
        )

        pipelines[name] = pipe
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV R² Mean": cv_scores.mean(),
            "CV R² Std": cv_scores.std()
        }

    return pipelines, metrics, X_test, y_test

models, performance_metrics, X_test, y_test = train_models(df)
selected_model = models[model_type]

# ---------------------------------------------------------
# Display Model Performance (Sidebar)
# ---------------------------------------------------------
with st.sidebar.expander("📈 Model Performance", expanded=True):
    perf = performance_metrics[model_type]
    st.metric("R² Score", f"{perf['R² Score']:.3f}")
    st.metric("MAE", f"${perf['MAE']:,.0f}")
    st.metric("RMSE", f"${perf['RMSE']:,.0f}")
    st.caption(f"CV: {perf['CV R² Mean']:.3f} ± {perf['CV R² Std']:.3f}")
    st.sidebar.success(f"Active Model: {model_type}")

# ---------------------------------------------------------
# PURE ML Prediction Helper
# ---------------------------------------------------------
def predict_salary(year: int, job: str, exp: str, size: str, model_pipeline: Pipeline):
    """
    Rules:
    - 2020–2022:
        If dataset has actual rows for (year, job, exp, size) → return actual avg (label: 'Actual')
        Otherwise → ML prediction (label: 'Predicted (ML)')
    - 2023–2030:
        Always ML prediction (label: 'Predicted (ML)')
    """
    # For years with historical data window
    if 2020 <= year <= 2022:
        mask = (
            (df["work_year"] == year) &
            (df["job_title"] == job) &
            (df["experience_level"] == exp) &
            (df["company_size"] == size)
        )
        subset = df[mask]
        if len(subset) > 0:
            actual_salary = subset["salary_in_usd"].mean()
            return float(actual_salary), "Actual"

    # Otherwise, ML prediction
    input_df = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size]
    })

    pred = model_pipeline.predict(input_df)[0]
    pred = max(0.0, float(pred))

    return pred, "Predicted (ML)"

# ---------------------------------------------------------
# Main Content - Tabs
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

    col1, col2, col3 = st.columns(3)
    with col1:
        custom_job = st.selectbox("👔 Job Title", sorted(df["job_title"].unique()))
    with col2:
        custom_exp = st.selectbox("📈 Experience Level", sorted(df["experience_level"].unique()))
    with col3:
        custom_size = st.selectbox("🏢 Company Size", sorted(df["company_size"].unique()))

    # Basic info about historical coverage for this profile
    profile_history = df[
        (df["job_title"] == custom_job) &
        (df["experience_level"] == custom_exp) &
        (df["company_size"] == custom_size)
    ]
    years_available = sorted(profile_history["work_year"].unique())

    if len(years_available) == 0:
        st.warning(
            "⚠️ No historical records in 2020–2022 for this exact profile "
            f"({custom_job}, {custom_exp}, {custom_size}).\n\n"
            "All years will be estimated using the selected ML model."
        )
    else:
        year_list_str = ", ".join(str(y) for y in years_available)
        st.info(
            f"ℹ️ Found **{len(profile_history)}** records for this profile "
            f"in years: **{year_list_str}**.\n\n"
            "- For any 2020–2022 year where this profile exists → uses actual average.\n"
            "- For 2023–2030 or missing years → uses ML predictions."
        )

    st.markdown("---")

    # Forecast 2020–2030
    all_years = np.arange(2020, 2031)
    forecast_rows = []

    for year in all_years:
        salary, source = predict_salary(
            year, custom_job, custom_exp, custom_size, selected_model
        )
        forecast_rows.append({
            "Year": year,
            "Salary (USD)": salary,
            "Source": source
        })

    forecast_df = pd.DataFrame(forecast_rows)

    # Catch negative (should not happen) or NaN
    if (forecast_df["Salary (USD)"] < 0).any():
        st.error("⚠️ Error: Negative salaries detected. Check model or data.")
    if forecast_df["Salary (USD)"].isna().any():
        st.error("⚠️ Error: NaN salaries detected. Some years could not be predicted.")

    # Split Actual vs Predicted
    actual_mask = forecast_df["Source"].str.contains("Actual", na=False)
    actual_df = forecast_df[actual_mask]
    predicted_df = forecast_df[~actual_mask]

    # Breakdown summary
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        actual_years = sorted(actual_df["Year"].tolist())
        if actual_years:
            st.success(
                f"✅ **Actual Data Years ({len(actual_years)}):** "
                f"{', '.join(map(str, actual_years))}"
            )
        else:
            st.warning("⚠️ No exact actual data for this profile in 2020–2022.")
    with col_info2:
        predicted_years = sorted(predicted_df["Year"].tolist())
        if predicted_years:
            preview = ", ".join(map(str, predicted_years[:5]))
            more = ", ..." if len(predicted_years) > 5 else ""
            st.info(
                f"🔮 **Predicted (ML) Years ({len(predicted_years)}):** "
                f"{preview}{more}"
            )
        else:
            st.warning("⚠️ No ML predictions generated.")

    # Metrics: 2020, 2025, 2030
    safe_df = forecast_df.dropna(subset=["Salary (USD)"])
    if not safe_df.empty:
        if 2020 in safe_df["Year"].values:
            start_salary = float(safe_df.loc[safe_df["Year"] == 2020, "Salary (USD)"].iloc[0])
        else:
            start_salary = float(safe_df.iloc[0]["Salary (USD)"])

        if 2030 in safe_df["Year"].values:
            end_salary = float(safe_df.loc[safe_df["Year"] == 2030, "Salary (USD)"].iloc[0])
        else:
            end_salary = float(safe_df.iloc[-1]["Salary (USD)"])

        if 2025 in safe_df["Year"].values:
            salary_2025 = float(safe_df.loc[safe_df["Year"] == 2025, "Salary (USD)"].iloc[0])
        else:
            salary_2025 = end_salary

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📅 2020 Salary", f"${start_salary:,.0f}")
        with col_b:
            st.metric("📅 2025 Salary (ML)", f"${salary_2025:,.0f}")
        with col_c:
            if start_salary > 0:
                total_growth = ((end_salary - start_salary) / start_salary) * 100
                st.metric("📈 Total Change (2020–2030)", f"{total_growth:.1f}%")
            else:
                st.metric("📈 Total Change (2020–2030)", "N/A")

    # Plot
    fig = go.Figure()

    if len(actual_df) > 0:
        fig.add_trace(go.Scatter(
            x=actual_df["Year"],
            y=actual_df["Salary (USD)"],
            mode="lines+markers",
            name="Actual Data (2020–2022)",
            line=dict(color="#10b981", width=5),
            marker=dict(size=12, color="#10b981", line=dict(width=2, color="white")),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.15)",
            hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> Actual<extra></extra>"
        ))

    if len(predicted_df) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_df["Year"],
            y=predicted_df["Salary (USD)"],
            mode="lines+markers",
            name="Predicted (ML)",
            line=dict(color="#667eea", width=5, dash="dash"),
            marker=dict(size=12, color="#764ba2", line=dict(width=2, color="white")),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.15)",
            hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> %{text}<extra></extra>",
            text=predicted_df["Source"]
        ))
    else:
        st.warning("⚠️ No predicted data to display for this profile.")

    if len(actual_df) > 0 and len(predicted_df) > 0:
        fig.add_vline(
            x=2022.5,
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text="Actual  |  ML Predictions",
            annotation_position="top"
        )

    fig.update_layout(
        title=dict(
            text=f"Salary Timeline (2020–2030): {custom_job}<br><sub>{custom_exp} | {custom_size}</sub>",
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

    # Detailed table
    with st.expander("📋 View Detailed Forecast Table"):
        display_df = forecast_df.copy()
        display_df["Salary (USD)"] = display_df["Salary (USD)"].apply(
            lambda v: f"${v:,.0f}" if pd.notnull(v) else "N/A"
        )

        st.caption("""
        **Data Sources:**
        - **Actual** → For 2020–2022, if this profile exists in the dataset, we use the real average salary.
        - **Predicted (ML)** → For 2023–2030, and for 2020–2022 years where this exact profile is missing,
          we use the selected machine learning model.
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
# TAB 2: Data Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution Analysis (Actual Data: 2020–2022)")

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
            title="Average Salary by Experience Level (Actual 2020–2022)",
            color="salary_in_usd",
            color_continuous_scale="Viridis",
            text="salary_in_usd"
        )
        fig_exp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
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
            title="Average Salary by Company Size (Actual 2020–2022)",
            color="salary_in_usd",
            color_continuous_scale="Plasma",
            text="salary_in_usd"
        )
        fig_size.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_size.update_layout(
            showlegend=False,
            xaxis_title="Company Size",
            yaxis_title="Avg Salary (USD)"
        )
        st.plotly_chart(fig_size, use_container_width=True)

    # Top paying jobs
    st.markdown("---")
    st.subheader("💎 Top 10 Highest Paying Jobs (Actual vs ML)")

    year_selector = st.radio(
        "Select Year",
        [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030],
        horizontal=True,
        index=2
    )

    if year_selector <= 2022:
        # Actual
        subset = df[df["work_year"] == year_selector]
        top_jobs = (
            subset.groupby("job_title")["salary_in_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        data_type = "Actual Data"
    else:
        # ML predictions
        all_jobs = df["job_title"].unique()
        rows = []
        for job in all_jobs:
            job_data = df[df["job_title"] == job]
            most_common_exp = job_data["experience_level"].mode()[0]
            most_common_size = job_data["company_size"].mode()[0]

            salary, _ = predict_salary(
                year_selector,
                job,
                most_common_exp,
                most_common_size,
                selected_model
            )
            rows.append({"job_title": job, "salary_in_usd": salary})

        top_jobs = (
            pd.DataFrame(rows)
            .sort_values("salary_in_usd", ascending=False)
            .head(10)
        )
        data_type = "ML Predictions"

    if len(top_jobs) > 0:
        fig_top = px.bar(
            top_jobs,
            x="salary_in_usd",
            y="job_title",
            orientation="h",
            title=f"Top 10 Highest Paying Jobs in {year_selector} ({data_type})",
            color="salary_in_usd",
            color_continuous_scale="Turbo",
            text="salary_in_usd"
        )
        fig_top.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
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

    salary_value, data_source = predict_salary(
        calc_year, custom_job, custom_exp, custom_size, selected_model
    )

    st.markdown("---")

    if data_source == "Actual":
        st.success("✅ Using **actual** dataset values for this year & profile.")
    else:
        st.info("🔮 Using **ML prediction** for this year & profile.")

    col_x, col_y, col_z = st.columns([1, 2, 1])
    with col_y:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Estimated Salary in {calc_year}</h3>
            <h1 style="font-size: 3.5rem; margin: 1rem 0;">${salary_value:,.0f}</h1>
            <p style="font-size: 1.1rem;">{custom_job}</p>
            <p>{custom_exp} | {custom_size}</p>
        </div>
        """, unsafe_allow_html=True)

    # Market comparison (historical job-level average, actual only)
    st.markdown("---")
    st.subheader("📊 Market Comparison (Job-Level)")

    market_avg_all = df[df["job_title"] == custom_job]["salary_in_usd"].mean()

    if calc_year <= 2022:
        market_year = df[
            (df["job_title"] == custom_job) &
            (df["work_year"] == calc_year)
        ]["salary_in_usd"].mean()
        if not np.isnan(market_year):
            market_avg = market_year
            market_label = f"Market Avg ({calc_year}, Actual)"
        else:
            market_avg = market_avg_all
            market_label = "Overall Market Avg (2020–2022, Actual)"
    else:
        market_avg = market_avg_all
        market_label = "Historical Market Avg (2020–2022, Actual)"

    diff = salary_value - market_avg
    diff_pct = (diff / market_avg) * 100 if market_avg > 0 else 0.0

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
        <h4>📌 How This Dashboard Works</h4>
        <ul>
            <li><b>Target:</b> <code>salary_in_usd</code> (continuous numeric)</li>
            <li><b>Features:</b> <code>work_year</code>, <code>job_title</code>, <code>experience_level</code>, <code>company_size</code></li>
            <li><b>Training Data:</b> Actual cybersecurity salaries from 2020–2022</li>
            <li><b>Predictions:</b>
                <ul>
                    <li><b>2020–2022</b>: Use actual averages when available for the profile; otherwise ML.</li>
                    <li><b>2023–2030</b>: Fully ML-based forecasts using the selected model.</li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model comparison
    st.markdown("### 📊 Model Comparison (on 2020–2022 Data)")

    comparison_data = []
    for name, metrics in performance_metrics.items():
        comparison_data.append({
            "Model": name,
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
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
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
        fig_mae.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("### 📈 Detailed Metrics")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Feature importance for tree-based models
    if model_type in ["Random Forest", "Gradient Boosting"]:
        st.markdown("---")
        st.subheader("🛠 Feature Importance")

        pipeline = models[model_type]  # Pipeline(prep, model)
        model_obj = pipeline.named_steps["model"]
        prep_obj = pipeline.named_steps["prep"]

        if hasattr(model_obj, "feature_importances_"):
            ohe = prep_obj.named_transformers_["cat"]
            importance_df = pd.DataFrame({
                "Importance": model_obj.feature_importances_
            })

            n_job = len(ohe.categories_[0])
            n_exp = len(ohe.categories_[1])
            n_size = len(ohe.categories_[2])

            feature_types = (
                ["Job Title"] * n_job +
                ["Experience"] * n_exp +
                ["Company Size"] * n_size +
                ["Year"]  # work_year passthrough
            )

            importance_df["Feature Type"] = feature_types

            importance_grouped = (
                importance_df.groupby("Feature Type")["Importance"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig_imp = px.bar(
                importance_grouped,
                x="Importance",
                y="Feature Type",
                orientation="h",
                title=f"Feature Importance – {model_type}",
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_imp, use_container_width=True)

            st.caption("💡 This shows which feature groups contribute most to the model's predictions.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><b>Built with Streamlit • Powered by Machine Learning</b></p>
        <p style='font-size: 0.9rem;'>
            Active Model: {model_type}
            &nbsp;|&nbsp;
            R²: {performance_metrics[model_type]["R² Score"]:.3f}
            &nbsp;|&nbsp;
            MAE: ${performance_metrics[model_type]["MAE"]:,.0f}
        </p>
        <p style='font-size: 0.8rem;'>
            Target: salary_in_usd &nbsp;|&nbsp;
            Features: work_year, job_title, experience_level, company_size
        </p>
    </div>
""", unsafe_allow_html=True)

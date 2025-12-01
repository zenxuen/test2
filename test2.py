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
st.markdown("*Actual Data: 2020–2022 | ML Predictions: 2023–2030*")
st.markdown("**Target:** `salary_in_usd`  &nbsp;&nbsp;|&nbsp;&nbsp; **Features:** `work_year`, `job_title`, `experience_level`, `employment_type`, `company_size`")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")

    # Keep only the columns we need
    keep_cols = [
        "work_year",
        "job_title",
        "experience_level",
        "employment_type",
        "company_size",
        "salary_in_usd"
    ]
    df = df[keep_cols].copy()

    # Basic cleaning
    df = df.dropna(subset=["salary_in_usd", "job_title",
                           "experience_level", "employment_type",
                           "company_size", "work_year"])

    # Ensure numeric types
    df["work_year"] = df["work_year"].astype(int)
    df["salary_in_usd"] = df["salary_in_usd"].astype(float)

    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar - Model Selection & Dataset Info
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Configuration")

model_type = st.sidebar.selectbox(
    "Select ML Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
    help="All predictions are made by this ML model."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
- **Total Records:** {len(df):,}
- **Available Years:** {", ".join(str(y) for y in sorted(df["work_year"].unique()))}
- **Unique Job Titles:** {df["job_title"].nunique()}
- **Unique Experience Levels:** {df["experience_level"].nunique()}
- **Unique Employment Types:** {df["employment_type"].nunique()}
- **Unique Company Sizes:** {df["company_size"].nunique()}
- **Avg Salary:** ${df['salary_in_usd'].mean():,.0f}
""")

# ---------------------------------------------------------
# Train ML Models (Target: salary_in_usd, Features as defined)
# ---------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    feature_cols = [
        "work_year",
        "job_title",
        "experience_level",
        "employment_type",
        "company_size"
    ]
    target_col = "salary_in_usd"

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = [
        "job_title",
        "experience_level",
        "employment_type",
        "company_size"
    ]
    # work_year is numeric, pass through
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough"  # keeps work_year
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
            max_depth=3,
            learning_rate=0.1
        )
    }

    trained = {}
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

        trained[name] = pipe
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV R² Mean": cv_scores.mean(),
            "CV R² Std": cv_scores.std()
        }

    return trained, metrics

models, performance_metrics = train_models(df)
selected_model = models[model_type]

# ---------------------------------------------------------
# Sidebar - Show Active Model Performance
# ---------------------------------------------------------
with st.sidebar.expander("📈 Active Model Performance", expanded=True):
    perf = performance_metrics[model_type]
    st.metric("R² Score", f"{perf['R² Score']:.3f}")
    st.metric("MAE", f"${perf['MAE']:,.0f}")
    st.metric("RMSE", f"${perf['RMSE']:,.0f}")
    st.caption(
        f"Cross-Validation R²: {perf['CV R² Mean']:.3f} ± {perf['CV R² Std']:.3f}"
    )

st.sidebar.success(f"✅ Active ML Model: {model_type}")

# ---------------------------------------------------------
# Helper: Get ML / Actual Salary for a Profile & Year
# ---------------------------------------------------------
def get_salary_for_profile(
    year: int,
    job: str,
    exp: str,
    emp_type: str,
    size: str,
    model
):
    """
    Rules:
    - 2020–2022:
        If there is actual data for (year, job, exp, emp_type, size):
            → use actual mean salary, Source="Actual"
        Else:
            → ML Prediction, Source="Predicted (ML)"
    - 2023–2030:
        → ML Prediction, Source="Predicted (ML)"
    """
    profile_rows = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["employment_type"] == emp_type) &
        (df["company_size"] == size)
    ]

    # Try to use actual data for 2020–2022
    if year <= 2022:
        actual_rows = profile_rows[profile_rows["work_year"] == year]
        if len(actual_rows) > 0:
            salary_actual = actual_rows["salary_in_usd"].mean()
            return float(salary_actual), "Actual"

    # Otherwise use ML prediction
    input_df = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "employment_type": [emp_type],
        "company_size": [size]
    })

    salary_pred = float(model.predict(input_df)[0])
    salary_pred = max(0.0, salary_pred)

    return salary_pred, "Predicted (ML)"

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
# TAB 1: Salary Forecast (Profile Selection + 2020–2030)
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Customize Your Profile")

    # Step 1: Job Title
    job_options = sorted(df["job_title"].unique())
    col_job, col_exp, col_emp, col_size = st.columns(4)

    with col_job:
        custom_job = st.selectbox("👔 Job Title", job_options)

    # Step 2: Experience options depend on Job
    exp_options = sorted(
        df[df["job_title"] == custom_job]["experience_level"].unique()
    )
    with col_exp:
        custom_exp = st.selectbox("📈 Experience Level", exp_options)

    # Step 3: Employment Type depends on (Job, Exp)
    emp_options = sorted(
        df[
            (df["job_title"] == custom_job) &
            (df["experience_level"] == custom_exp)
        ]["employment_type"].unique()
    )
    with col_emp:
        custom_emp_type = st.selectbox("💼 Employment Type", emp_options)

    # Step 4: Company Size depends on (Job, Exp, EmpType)
    size_options = sorted(
        df[
            (df["job_title"] == custom_job) &
            (df["experience_level"] == custom_exp) &
            (df["employment_type"] == custom_emp_type)
        ]["company_size"].unique()
    )
    with col_size:
        custom_size = st.selectbox("🏢 Company Size", size_options)

    # Quick info about how many rows this profile has
    profile_rows = df[
        (df["job_title"] == custom_job) &
        (df["experience_level"] == custom_exp) &
        (df["employment_type"] == custom_emp_type) &
        (df["company_size"] == custom_size)
    ]
    years_available = sorted(profile_rows["work_year"].unique())
    if len(years_available) > 0:
        st.info(
            f"📚 This profile has **{len(profile_rows)} records** "
            f"across years: {', '.join(str(y) for y in years_available)}."
        )
    else:
        # 理论上不会发生，因为我们已经限制下拉选项了
        st.warning(
            "⚠️ This profile has no direct historical records. "
            "All values shown will come purely from the ML model."
        )

    st.markdown("---")

    # Forecast 2020–2030
    all_years = list(range(2020, 2031))
    forecast_data = []
    for yr in all_years:
        sal, src = get_salary_for_profile(
            yr, custom_job, custom_exp, custom_emp_type, custom_size, selected_model
        )
        forecast_data.append({
            "Year": yr,
            "Salary (USD)": sal,
            "Source": src
        })

    forecast_df = pd.DataFrame(forecast_data)

    # Split Actual vs ML
    actual_mask = forecast_df["Source"] == "Actual"
    actual_df = forecast_df[actual_mask]
    predicted_df = forecast_df[~actual_mask]

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        if len(actual_df) > 0:
            st.success(
                "✅ **Actual Data Years:** " +
                ", ".join(str(y) for y in actual_df["Year"].tolist())
            )
        else:
            st.warning("⚠️ No exact actual data for this profile in 2020–2022.")

    with col_info2:
        pred_years = predicted_df["Year"].tolist()
        if len(pred_years) > 0:
            st.info(
                "🔮 **ML Prediction Years:** " +
                ", ".join(str(y) for y in pred_years)
            )
        else:
            st.warning("⚠️ No ML predictions generated.")

    # Basic metrics: 2020 vs 2030 (if available)
    safe_df = forecast_df.dropna(subset=["Salary (USD)"])
    if not safe_df.empty:
        # try to get 2020 & 2030, else first & last
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

        mid_year = 2025
        if mid_year in safe_df["Year"].values:
            mid_salary = float(
                safe_df[safe_df["Year"] == mid_year]["Salary (USD)"].iloc[0]
            )
        else:
            mid_salary = end_salary

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📅 2020 Salary", f"${start_salary:,.0f}")
        with col_b:
            st.metric(f"📅 {mid_year} Salary (ML)", f"${mid_salary:,.0f}")
        with col_c:
            if start_salary > 0:
                growth = (end_salary - start_salary) / start_salary * 100
                st.metric("📈 Total Growth (2020–2030)", f"{growth:.1f}%")
            else:
                st.metric("📈 Total Growth (2020–2030)", "N/A")

    # Plotly visualization
    fig = go.Figure()

    # Actual
    if len(actual_df) > 0:
        fig.add_trace(go.Scatter(
            x=actual_df["Year"],
            y=actual_df["Salary (USD)"],
            mode="lines+markers",
            name="Actual Data (2020–2022)",
            line=dict(color="#10b981", width=4),
            marker=dict(size=10, color="#10b981",
                        line=dict(width=1, color="white")),
            hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<extra></extra>"
        ))

    # Predicted (ML)
    if len(predicted_df) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_df["Year"],
            y=predicted_df["Salary (USD)"],
            mode="lines+markers",
            name="Predicted (ML)",
            line=dict(color="#667eea", width=4, dash="dash"),
            marker=dict(size=10, color="#764ba2",
                        line=dict(width=1, color="white")),
            hovertemplate="<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> %{text}<extra></extra>",
            text=predicted_df["Source"]
        ))

    # Vertical line at 2022.5 if both exist
    if len(actual_df) > 0 and len(predicted_df) > 0:
        fig.add_vline(
            x=2022.5,
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text="Actual | ML Predictions",
            annotation_position="top"
        )

    fig.update_layout(
        title=dict(
            text=(
                f"Salary Forecast (2020–2030): {custom_job}"
                f"<br><sub>{custom_exp} | {custom_emp_type} | {custom_size}</sub>"
            ),
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

    st.plotly_chart(fig, width="stretch")

    # Detailed table
    with st.expander("📋 View Detailed Forecast Table"):
        display_df = forecast_df.copy()
        display_df["Salary (USD)"] = display_df["Salary (USD)"].apply(
            lambda v: f"${v:,.0f}" if pd.notnull(v) else "N/A"
        )
        st.caption("""
        **Data Sources:**
        - **Actual:** Directly from dataset (2020–2022, if available for this profile).
        - **Predicted (ML):** Output from the chosen ML model using
          `work_year`, `job_title`, `experience_level`, `employment_type`, `company_size`.
        """)
        st.dataframe(display_df, width="stretch", hide_index=True)

# ---------------------------------------------------------
# TAB 2: Data Analysis (Actual + ML)
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution Analysis (Actual Data 2020–2022)")

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
        fig_exp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_exp.update_layout(
            showlegend=False,
            xaxis_title="Experience Level",
            yaxis_title="Average Salary (USD)",
            height=450
        )
        st.plotly_chart(fig_exp, width="stretch")

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
        fig_size.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_size.update_layout(
            showlegend=False,
            xaxis_title="Company Size",
            yaxis_title="Average Salary (USD)",
            height=450
        )
        st.plotly_chart(fig_size, width="stretch")

    st.markdown("---")
    st.subheader("💎 Top 10 Highest Paying Jobs by Year")

    year_selector = st.radio(
        "Select Year",
        list(range(2020, 2031)),
        index=2,
        horizontal=True
    )

    # For each job, use the most common (exp, emp_type, size)
    top_data = []
    for job in df["job_title"].unique():
        job_df = df[df["job_title"] == job]

        most_exp = job_df["experience_level"].mode()[0]
        most_emp_type = job_df["employment_type"].mode()[0]
        most_size = job_df["company_size"].mode()[0]

        salary_val, src = get_salary_for_profile(
            year_selector, job, most_exp, most_emp_type, most_size, selected_model
        )

        top_data.append({
            "job_title": job,
            "salary_in_usd": salary_val
        })

    top_jobs = (
        pd.DataFrame(top_data)
        .sort_values("salary_in_usd", ascending=False)
        .head(10)
    )

    if year_selector <= 2022:
        data_type = "Actual/ML Mixed (Actual where available)"
    else:
        data_type = "ML Predictions"

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
        height=550,
        xaxis_title="Salary (USD)",
        yaxis_title=""
    )
    st.plotly_chart(fig_top, width="stretch")

# ---------------------------------------------------------
# TAB 3: Salary Calculator
# ---------------------------------------------------------
with tab3:
    st.subheader("💰 Salary Calculator")

    # Reuse the same profile from Tab 1
    calc_year = st.slider("📅 Select Year", 2020, 2030, 2025)

    salary_val, src = get_salary_for_profile(
        calc_year, custom_job, custom_exp, custom_emp_type, custom_size, selected_model
    )

    st.markdown("---")

    if src == "Actual":
        st.success(f"✅ Source: {src} (from dataset)")
    else:
        st.info(f"🔮 Source: {src}")

    col_x, col_y, col_z = st.columns([1, 2, 1])
    with col_y:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Estimated Salary in {calc_year}</h3>
            <h1 style="font-size: 3.5rem; margin: 1rem 0;">${salary_val:,.0f}</h1>
            <p style="font-size: 1.1rem;">{custom_job}</p>
            <p>{custom_exp} | {custom_emp_type} | {custom_size}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Market Context (Historical)")

    # Market average for this job (historical only)
    market_avg_all = df[df["job_title"] == custom_job]["salary_in_usd"].mean()

    market_label = "Historical Avg (2020–2022)"
    market_avg = market_avg_all

    diff = salary_val - market_avg
    diff_pct = (diff / market_avg) * 100 if market_avg > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Salary (This Profile)", f"${salary_val:,.0f}")
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
        <h4>📌 How This ML Model Works</h4>
        <ul>
            <li><b>Target (y):</b> <code>salary_in_usd</code> — the annual salary in USD</li>
            <li><b>Features (X):</b>
                <ul>
                    <li><code>work_year</code> — the calendar year (for capturing time trend)</li>
                    <li><code>job_title</code> — role in cybersecurity (e.g. Security Engineer, Analyst)</li>
                    <li><code>experience_level</code> — junior / mid / senior / expert, etc.</li>
                    <li><code>employment_type</code> — FT, PT, contract, etc.</li>
                    <li><code>company_size</code> — S / M / L</li>
                </ul>
            </li>
            <li><b>Algorithm:</b> You can switch between Linear Regression, Random Forest, and Gradient Boosting in the sidebar.</li>
            <li><b>Predictions:</b> For 2023–2030, all values come from the chosen ML model.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Comparison")

    comparison_data = []
    for name, m in performance_metrics.items():
        comparison_data.append({
            "Model": name,
            "R² Score": m["R² Score"],
            "MAE ($)": m["MAE"],
            "RMSE ($)": m["RMSE"]
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
        fig_r2.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_r2, width="stretch")

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
        fig_mae.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_mae, width="stretch")

    st.markdown("### 🎯 Feature Importance (by Feature Group)")

    pipeline = models[model_type]
    model_obj = pipeline.named_steps["model"]
    prep_obj = pipeline.named_steps["prep"]

    if hasattr(prep_obj, "named_transformers_") and "cat" in prep_obj.named_transformers_:
        ohe = prep_obj.named_transformers_["cat"]
        n_job = len(ohe.categories_[0])
        n_exp = len(ohe.categories_[1])
        n_emp = len(ohe.categories_[2])
        n_size = len(ohe.categories_[3])
        # +1 for work_year (numeric passthrough)
        feature_types = (
            ["Job Title"] * n_job +
            ["Experience Level"] * n_exp +
            ["Employment Type"] * n_emp +
            ["Company Size"] * n_size +
            ["Work Year"]
        )

        # Get raw importance vector
        if hasattr(model_obj, "feature_importances_"):
            raw_imp = np.array(model_obj.feature_importances_)
        elif hasattr(model_obj, "coef_"):
            raw_imp = np.abs(np.array(model_obj.coef_))
        else:
            raw_imp = None

        if raw_imp is not None and len(raw_imp) == len(feature_types):
            imp_df = pd.DataFrame({
                "Feature Group": feature_types,
                "Importance": raw_imp
            })
            grouped_imp = (
                imp_df.groupby("Feature Group")["Importance"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig_imp = px.bar(
                grouped_imp,
                x="Importance",
                y="Feature Group",
                orientation="h",
                title=f"Relative Importance by Feature Group — {model_type}",
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig_imp.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=450,
                showlegend=False
            )
            st.plotly_chart(fig_imp, width="stretch")
            st.caption("💡 Bars further to the right indicate feature groups that the model relies on more when predicting salary.")
        else:
            st.info("ℹ️ Feature importance is not available for this model.")
    else:
        st.info("ℹ️ Feature importance cannot be extracted for this configuration.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><b>Built with Streamlit • Powered purely by Machine Learning</b></p>
        <p style='font-size: 0.9rem;'>
            Active Model: {model_type} |
            R² Score: {performance_metrics[model_type]["R² Score"]:.3f} |
            MAE: ${performance_metrics[model_type]["MAE"]:,.0f}
        </p>
        <p style='font-size: 0.8rem;'>
            Target: <code>salary_in_usd</code> |
            Features: <code>work_year</code>, <code>job_title</code>,
            <code>experience_level</code>, <code>employment_type</code>, <code>company_size</code>
        </p>
    </div>
""", unsafe_allow_html=True)

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

# Prediction method selector
prediction_method = st.sidebar.radio(
    "Prediction Method",
    ["Growth-Based (Recommended)", "Pure ML Model", "Hybrid"],
    help="Growth-Based: Uses historical growth rates\nPure ML: Uses model only\nHybrid: Blends both approaches"
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

st.sidebar.success(f"**Active Model:** {model_type}")

# ---------------------------------------------------------
# Train Models - Target: salary_in_usd, Predictors: All others
# ---------------------------------------------------------
@st.cache_resource
def train_models(data):
    # TARGET: salary_in_usd
    # PREDICTORS: work_year, job_title, experience_level, company_size
    X = data[["work_year", "job_title", "experience_level", "company_size"]]
    y = data["salary_in_usd"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Categorical columns to encode
    categorical_cols = ["job_title", "experience_level", "company_size"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"  # Keep work_year as numeric
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
            max_depth=7, 
            learning_rate=0.1
        )
    }
    
    trained_models = {}
    metrics = {}
    
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
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
# Calculate Growth Rates from Historical Data
# ---------------------------------------------------------
@st.cache_data
def calculate_growth_rate(data, job, exp, size):
    """Calculate historical growth rate for this specific profile"""
    # Try to get data for this specific combination
    profile_data = data[
        (data["job_title"] == job) & 
        (data["experience_level"] == exp) & 
        (data["company_size"] == size)
    ].groupby("work_year")["salary_in_usd"].mean()
    
    if len(profile_data) >= 2:
        # Has multiple years - calculate actual growth
        years = profile_data.index.values
        salaries = profile_data.values
        growth = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        return max(0.03, min(growth, 0.15)), "profile"  # Cap between 3-15%
    
    # Only 1 year or no data - try job level
    job_data = data[data["job_title"] == job].groupby("work_year")["salary_in_usd"].mean()
    if len(job_data) >= 2:
        years = job_data.index.values
        salaries = job_data.values
        growth = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        return max(0.03, min(growth, 0.12)), "job"
    
    # Try experience level
    exp_data = data[data["experience_level"] == exp].groupby("work_year")["salary_in_usd"].mean()
    if len(exp_data) >= 2:
        years = exp_data.index.values
        salaries = exp_data.values
        growth = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        return max(0.03, min(growth, 0.12)), "experience"
    
    # Default to market average (5% annual growth)
    return 0.05, "market"

# ---------------------------------------------------------
# Get mean prediction from similar profiles
# ---------------------------------------------------------
@st.cache_data
def get_mean_prediction_from_similar(data, year, job, exp, _model):
    """
    If a profile has only 1 year of data, predict based on mean of similar profiles
    """
    # Get all profiles with the same job and experience (ignore company size)
    similar_profiles = data[
        (data["job_title"] == job) & 
        (data["experience_level"] == exp)
    ]
    
    if len(similar_profiles) == 0:
        # No similar profiles, use job-only
        similar_profiles = data[data["job_title"] == job]
    
    if len(similar_profiles) == 0:
        # Still nothing, use experience-only
        similar_profiles = data[data["experience_level"] == exp]
    
    if len(similar_profiles) == 0:
        # No similar profiles at all
        return None
    
    # Get the latest year's average salary for this profile
    latest_year = similar_profiles['work_year'].max()
    base_salary = similar_profiles[similar_profiles['work_year'] == latest_year]['salary_in_usd'].mean()
    
    # Calculate growth rate from similar profiles
    yearly_avg = similar_profiles.groupby('work_year')['salary_in_usd'].mean()
    if len(yearly_avg) >= 2:
        years = yearly_avg.index.values
        salaries = yearly_avg.values
        growth = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        growth = max(0.03, min(growth, 0.15))  # Cap between 3-15%
    else:
        growth = 0.05  # Default 5%
    
    # Project from latest year
    years_ahead = year - latest_year
    predicted = base_salary * ((1 + growth) ** years_ahead)
    
    return max(0, predicted)  # Ensure non-negative

# ---------------------------------------------------------
# Prediction Function with Multiple Methods
# ---------------------------------------------------------
def get_salary(year, job, exp, size, method="Growth-Based (Recommended)"):
    """
    Get salary with multiple prediction methods
    - For profiles with only 1 year: Use mean of similar profiles' predictions
    """
    if year >= 2020 and year <= 2022:
        # USE ACTUAL DATA
        actual = df[
            (df["work_year"] == year) & 
            (df["job_title"] == job) & 
            (df["experience_level"] == exp) & 
            (df["company_size"] == size)
        ]
        
        if len(actual) > 0:
            return actual["salary_in_usd"].mean(), "Actual"
        else:
            actual_partial = df[
                (df["work_year"] == year) & 
                (df["job_title"] == job) & 
                (df["experience_level"] == exp)
            ]
            
            if len(actual_partial) > 0:
                return actual_partial["salary_in_usd"].mean(), "Actual (Partial)"
            else:
                actual_job = df[
                    (df["work_year"] == year) & 
                    (df["job_title"] == job)
                ]
                
                if len(actual_job) > 0:
                    return actual_job["salary_in_usd"].mean(), "Actual (Job Only)"
    
    # PREDICTIONS for 2023-2030 (or 2020-2022 if no actual data exists)
    
    # Get profile history
    profile_history = df[
        (df["job_title"] == job) & 
        (df["experience_level"] == exp) & 
        (df["company_size"] == size)
    ]
    
    # If NO data at all for this profile, try broader match
    if len(profile_history) == 0:
        profile_history = df[
            (df["job_title"] == job) & 
            (df["experience_level"] == exp)
        ]
    
    if len(profile_history) == 0:
        profile_history = df[df["job_title"] == job]
    
    if len(profile_history) == 0:
        # Absolutely no data - use overall market average
        market_avg = df.groupby('work_year')['salary_in_usd'].mean()
        if len(market_avg) >= 2:
            years = market_avg.index.values
            salaries = market_avg.values
            growth = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
            growth = max(0.03, min(growth, 0.12))
        else:
            growth = 0.05
        
        base_salary = df['salary_in_usd'].mean()
        base_year = df['work_year'].max()
        years_ahead = year - base_year
        return max(0, base_salary * ((1 + growth) ** years_ahead)), "Predicted (Market Avg)"
    
    unique_years = profile_history["work_year"].nunique()
    
    # If only 1 year of data, use mean of similar profiles
    if unique_years == 1:
        mean_pred = get_mean_prediction_from_similar(df, year, job, exp, selected_model)
        
        if mean_pred is not None and mean_pred > 0:
            return mean_pred, "Predicted (Mean of Similar)"
        # If that fails, continue to other methods
    
    # Method 1: Growth-Based (Most Realistic)
    if method == "Growth-Based (Recommended)":
        if len(profile_history) > 0:
            last_year_data = profile_history[profile_history["work_year"] == profile_history["work_year"].max()]
            base_salary = last_year_data["salary_in_usd"].mean()
            base_year = last_year_data["work_year"].iloc[0]
            
            # Ensure base salary is positive
            if base_salary <= 0:
                base_salary = df[df['job_title'] == job]['salary_in_usd'].mean()
            
            growth_rate, growth_source = calculate_growth_rate(df, job, exp, size)
            years_ahead = year - base_year
            predicted_salary = base_salary * ((1 + growth_rate) ** years_ahead)
            
            return max(0, predicted_salary), f"Predicted ({growth_source})"
    
    # Method 2: Pure ML Model
    elif method == "Pure ML Model":
        pred_input = pd.DataFrame({
            "work_year": [year],
            "job_title": [job],
            "experience_level": [exp],
            "company_size": [size]
        })
        predicted_salary = selected_model.predict(pred_input)[0]
        return max(0, predicted_salary), "Predicted"
    
    # Method 3: Hybrid
    else:
        if len(profile_history) > 0:
            last_year_data = profile_history[profile_history["work_year"] == profile_history["work_year"].max()]
            base_salary = last_year_data["salary_in_usd"].mean()
            base_year = last_year_data["work_year"].iloc[0]
            
            if base_salary <= 0:
                base_salary = df[df['job_title'] == job]['salary_in_usd'].mean()
            
            growth_rate, _ = calculate_growth_rate(df, job, exp, size)
            years_ahead = year - base_year
            growth_prediction = base_salary * ((1 + growth_rate) ** years_ahead)
            
            # Get ML prediction
            pred_input = pd.DataFrame({
                "work_year": [year],
                "job_title": [job],
                "experience_level": [exp],
                "company_size": [size]
            })
            ml_prediction = selected_model.predict(pred_input)[0]
            
            # Blend: 70% growth-based, 30% ML
            blended = 0.7 * growth_prediction + 0.3 * ml_prediction
            return max(0, blended), "Predicted"
    
    # Final fallback
    pred_input = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size]
    })
    predicted_salary = selected_model.predict(pred_input)[0]
    return max(0, predicted_salary), "Predicted"

# ---------------------------------------------------------
# Main Content - Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Salary Forecast", "📊 Data Analysis", "💰 Salary Calculator", "🗺️ Model Insights"])

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
    
    # Show growth rate info
    if prediction_method == "Growth-Based (Recommended)":
        # Check if profile has only 1 year
        profile_history = df[
            (df["job_title"] == custom_job) & 
            (df["experience_level"] == custom_exp) & 
            (df["company_size"] == custom_size)
        ]
        unique_years = profile_history["work_year"].nunique()
        
        if unique_years == 1:
            st.warning(f"⚠️ This profile has only **1 year** of data ({profile_history['work_year'].iloc[0]}). Predictions use **mean of similar profiles** (same job + experience).")
        elif unique_years >= 2:
            growth_rate, growth_source = calculate_growth_rate(df, custom_job, custom_exp, custom_size)
            years_list = sorted(profile_history['work_year'].unique())
            st.success(f"✅ This profile has **{unique_years} years** of data ({', '.join(map(str, years_list))}). Using **its own growth pattern**: **{growth_rate*100:.1f}% per year**")
        else:
            st.info(f"ℹ️ No historical data for this exact profile. Using fallback prediction methods.")
    
    st.markdown("---")
    
    # Generate forecast for 2020-2030
    all_years = np.arange(2020, 2031)
    forecast_data = []
    
    for year in all_years:
        salary, source = get_salary(year, custom_job, custom_exp, custom_size, prediction_method)
        forecast_data.append({
            "Year": year,
            "Salary (USD)": salary,
            "Source": source
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Validate: Check for negative salaries
    if (forecast_df["Salary (USD)"] < 0).any():
        st.error("⚠️ **Error**: Negative salaries detected! This profile may have insufficient data. Try selecting a different combination.")
        negative_years = forecast_df[forecast_df["Salary (USD)"] < 0]["Year"].tolist()
        st.warning(f"Years with negative predictions: {negative_years}")
    
    # Debug: Show what data we have
    actual_count = len(forecast_df[forecast_df["Source"].str.contains("Actual", na=False)])
    predicted_count = len(forecast_df[forecast_df["Source"] == "Predicted"])
    
    # Show data breakdown - only show actual years that are actually actual
    actual_years = forecast_df[forecast_df["Source"].str.contains("Actual", na=False)]["Year"].tolist()
    # Predicted years should be 2023+ or any year without actual data
    predicted_years = [y for y in forecast_df["Year"].tolist() if y not in actual_years and y >= 2020]
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        if actual_years:
            st.success(f"✅ **Actual Data ({actual_count} years):** {', '.join(map(str, actual_years))}")
        else:
            st.warning("⚠️ No actual data found for this combination")
    with col_info2:
        if predicted_years:
            st.info(f"🔮 **Predicted Data ({predicted_count} years):** {', '.join(map(str, predicted_years))}")
        else:
            st.warning("⚠️ No predictions generated")
    
    # Calculate metrics
    start_salary = forecast_df.iloc[0]['Salary (USD)']
    end_salary = forecast_df.iloc[-1]['Salary (USD)']
    salary_2025 = forecast_df[forecast_df["Year"] == 2025].iloc[0]['Salary (USD)']
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("📅 2020 Salary", f"${start_salary:,.0f}")
    with col_b:
        st.metric("📅 2025 Salary (Predicted)", f"${salary_2025:,.0f}")
    with col_c:
        growth = ((end_salary - start_salary) / start_salary) * 100
        st.metric("📈 Total Growth (2020-2030)", f"{growth:.1f}%")
    
    # Enhanced Visualization
    fig = go.Figure()
    
    # Plot Actual Data (2020-2022)
    actual_df = forecast_df[forecast_df["Source"].str.contains("Actual", na=False)]
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
    
    # Plot Predicted Data (2023-2030)
    predicted_df = forecast_df[forecast_df["Source"] == "Predicted"]
    if len(predicted_df) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_df["Year"],
            y=predicted_df["Salary (USD)"],
            mode='lines+markers',
            name='ML Predictions (2023-2030)',
            line=dict(color='#667eea', width=5, dash='dash'),
            marker=dict(size=14, color='#764ba2', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.15)',
            hovertemplate='<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> Predicted<extra></extra>'
        ))
    else:
        st.warning("⚠️ No predicted data to display. Check if there's historical data for this profile.")
    
    # Add vertical separator only if we have both actual and predicted
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
    
    # Show detailed table
    with st.expander("📋 View Detailed Forecast Table"):
        display_df = forecast_df.copy()
        display_df['Salary (USD)'] = display_df['Salary (USD)'].apply(lambda x: f"${x:,.0f}")
        
        # Add explanation for different source types
        st.caption("""
        **Data Sources:**
        - **Actual**: Real data from dataset
        - **Predicted (Mean of Similar)**: Only 1 year available, using average of similar job+experience combinations
        - **Predicted (profile/job/experience/market)**: Growth-based prediction using respective level data
        """)
        
        # Color code the source
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
        fig_exp.update_layout(showlegend=False, xaxis_title="Experience Level", yaxis_title="Avg Salary (USD)")
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
        fig_size.update_layout(showlegend=False, xaxis_title="Company Size", yaxis_title="Avg Salary (USD)")
        st.plotly_chart(fig_size, use_container_width=True)
    
    # Top Paying Jobs
    st.markdown("---")
    st.subheader("💎 Top 10 Highest Paying Jobs")
    
    year_selector = st.radio(
        "Select Year",
        [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030],
        horizontal=True,
        index=2  # Default 2022
    )
    
    # Get top jobs for selected year
    if year_selector <= 2022:
        # Use actual data
        top_jobs = df[df['work_year'] == year_selector].groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10).reset_index()
        data_type = "Actual Data"
    else:
        # Use predictions
        all_jobs = df['job_title'].unique()
        job_salaries = []
        
        for job in all_jobs:
            job_data = df[df['job_title'] == job]
            # Use most common experience and company size
            most_common_exp = job_data['experience_level'].mode()[0]
            most_common_size = job_data['company_size'].mode()[0]
            
            salary, _ = get_salary(year_selector, job, most_common_exp, most_common_size, prediction_method)
            job_salaries.append({
                "job_title": job,
                "salary_in_usd": salary
            })
        
        top_jobs = pd.DataFrame(job_salaries).sort_values('salary_in_usd', ascending=False).head(10)
        data_type = "ML Predictions"
    
    if len(top_jobs) > 0:
        fig_top = px.bar(
            top_jobs,
            x="salary_in_usd",
            y="job_title",
            orientation='h',
            title=f"Top 10 Highest Paying Jobs in {year_selector} ({data_type})",
            color="salary_in_usd",
            color_continuous_scale="Turbo",
            text="salary_in_usd"
        )
        fig_top.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_top.update_layout(
            yaxis={'categoryorder':'total ascending'}, 
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
    
    # Get salary
    salary_value, data_source = get_salary(calc_year, custom_job, custom_exp, custom_size, prediction_method)
    
    st.markdown("---")
    
    # Display data source
    if "Actual" in data_source:
        st.success(f"✅ Using {data_source} from dataset")
    else:
        st.info(f"🔮 Using ML Model Prediction ({model_type})")
    
    # Display salary
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
    
    # Market comparison
    st.markdown("---")
    st.subheader("📊 Market Comparison")
    
    # Calculate market average for this job across all years
    market_avg_all = df[df["job_title"] == custom_job]["salary_in_usd"].mean()
    
    # Calculate for specific year if actual data exists
    if calc_year <= 2022:
        market_year = df[(df["job_title"] == custom_job) & (df["work_year"] == calc_year)]["salary_in_usd"].mean()
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
            <li><b>Predictions:</b> Model forecasts for 2023-2030 based on learned patterns</li>
            <li><b>Model learns:</b> How year, job type, experience, and company size affect salary</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model comparison
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
    
    # Show detailed metrics table
    st.markdown("### 📈 Detailed Metrics")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Feature importance (if Random Forest or Gradient Boosting)
    if model_type in ["Random Forest", "Gradient Boosting"]:
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance")
        
        # Get feature importance
        model_estimator = selected_model.named_steps['model']
        if hasattr(model_estimator, 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = selected_model.named_steps['prep']
            
            # Get categorical feature names
            cat_features = []
            ohe = preprocessor.named_transformers_['cat']
            for i, cat in enumerate(["job_title", "experience_level", "company_size"]):
                cat_features.extend([f"{cat}_{val}" for val in ohe.categories_[i]])
            
            # Add numeric feature
            cat_features.append("work_year")
            
            importance_df = pd.DataFrame({
                'Feature Type': ['Year'] + ['Job Title']*len(ohe.categories_[0]) + 
                               ['Experience']*len(ohe.categories_[1]) + 
                               ['Company Size']*len(ohe.categories_[2]),
                'Importance': model_estimator.feature_importances_
            })
            
            # Group by feature type
            importance_grouped = importance_df.groupby('Feature Type')['Importance'].sum().sort_values(ascending=False).reset_index()
            
            fig_imp = px.bar(
                importance_grouped,
                x='Importance',
                y='Feature Type',
                orientation='h',
                title=f"Feature Importance - {model_type}",
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.caption("💡 This shows which factors have the most impact on salary predictions")

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

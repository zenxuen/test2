import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import plotly.express as px

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("📈 Cybersecurity Salary Predictor (2021 - 2025)")

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

st.success("Dataset loaded successfully!")

# ---------------------------------------------------------
# Features & Target
# ---------------------------------------------------------
X = df[['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']]
y = df['salary_in_usd']

# ---------------------------------------------------------
# Preprocessing with Random Forest
# ---------------------------------------------------------
categorical_features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
])

model.fit(X, y)
st.success("Random Forest Model trained successfully!")

# ---------------------------------------------------------
# Salary Calculator Section
# ---------------------------------------------------------
st.header("💰 Salary Calculator")
st.markdown("Select your profile details to get a salary prediction:")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    calc_work_year = st.selectbox("📅 Work Year", sorted(df['work_year'].unique()), key="calc_year")
    calc_experience = st.selectbox("📈 Experience Level", sorted(df['experience_level'].unique()), key="calc_exp")
    calc_employment = st.selectbox("💼 Employment Type", sorted(df['employment_type'].unique()), key="calc_emp")

with col2:
    calc_job_title = st.selectbox("👔 Job Title", sorted(df['job_title'].unique()), key="calc_job")
    calc_residence = st.selectbox("🏠 Employee Residence", sorted(df['employee_residence'].unique()), key="calc_res")
    calc_remote = st.slider("🌐 Remote Ratio (%)", 0, 100, 50, key="calc_remote")

with col3:
    calc_location = st.selectbox("🌍 Company Location", sorted(df['company_location'].unique()), key="calc_loc")
    calc_size = st.selectbox("🏢 Company Size", sorted(df['company_size'].unique()), key="calc_size")

# Calculate prediction
calc_input = pd.DataFrame([{
    'work_year': calc_work_year,
    'experience_level': calc_experience,
    'employment_type': calc_employment,
    'job_title': calc_job_title,
    'employee_residence': calc_residence,
    'remote_ratio': calc_remote,
    'company_location': calc_location,
    'company_size': calc_size
}])

calc_predicted_salary = model.predict(calc_input)[0]

# Display result with styling
st.markdown("---")
result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

with result_col2:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
        <h3 style='margin: 0; font-size: 1.2rem;'>Predicted Salary for {calc_work_year}</h3>
        <h1 style='margin: 1rem 0; font-size: 3.5rem;'>${calc_predicted_salary:,.0f}</h1>
        <p style='margin: 0; opacity: 0.9;'>{calc_job_title} | {calc_experience} | {calc_size}</p>
    </div>
    """, unsafe_allow_html=True)

# Market comparison
st.markdown("---")
st.subheader("📊 Market Comparison")

# Get market average for the selected job title and year
market_data = df[(df['job_title'] == calc_job_title) & (df['work_year'] == calc_work_year)]
if len(market_data) > 0:
    market_avg = market_data['salary_in_usd'].mean()
    market_label = f"Market Avg for {calc_job_title} ({calc_work_year})"
else:
    market_avg = df[df['job_title'] == calc_job_title]['salary_in_usd'].mean()
    market_label = f"Overall Market Avg for {calc_job_title}"

diff = calc_predicted_salary - market_avg
diff_pct = (diff / market_avg) * 100 if market_avg > 0 else 0

comp_col1, comp_col2, comp_col3 = st.columns(3)
with comp_col1:
    st.metric("Your Prediction", f"${calc_predicted_salary:,.0f}")
with comp_col2:
    st.metric(market_label, f"${market_avg:,.0f}")
with comp_col3:
    st.metric("Difference", f"${diff:,.0f}", f"{diff_pct:+.1f}%")

# ---------------------------------------------------------
# Visualization Section
# ---------------------------------------------------------
st.markdown("---")
st.header("📊 Data Visualization")

# Interactive selection for visualization
st.sidebar.header("🎨 Visualization Filters")
viz_type = st.sidebar.radio("Visualization Type", ["Box Plot by Year", "Scatter Plot", "Salary Trends"])
viz_job_title = st.sidebar.selectbox("Filter by Job Title", ["All"] + sorted(df['job_title'].unique().tolist()))

# Filter data if specific job selected
if viz_job_title != "All":
    df_viz = df[df['job_title'] == viz_job_title].copy()
else:
    df_viz = df.copy()

if viz_type == "Box Plot by Year":
    # Box plot shows distribution better
    fig = px.box(
        df_viz,
        x='work_year',
        y='salary_in_usd',
        color='experience_level',
        title=f"Salary Distribution by Year {f'for {viz_job_title}' if viz_job_title != 'All' else '(All Jobs)'}",
        labels={'salary_in_usd': 'Salary (USD)', 'work_year': 'Year', 'experience_level': 'Experience'},
        hover_data=['job_title', 'company_size']
    )
    fig.update_layout(height=600)
    
elif viz_type == "Scatter Plot":
    # Scatter plot with jitter
    import numpy as np
    df_viz_scatter = df_viz.copy()
    # Add small random jitter to x-axis to prevent overlap
    df_viz_scatter['work_year_jitter'] = df_viz_scatter['work_year'] + np.random.uniform(-0.2, 0.2, len(df_viz_scatter))
    
    fig = px.scatter(
        df_viz_scatter,
        x='work_year_jitter',
        y='salary_in_usd',
        color='experience_level',
        size='remote_ratio',
        hover_data=['experience_level', 'company_size', 'job_title', 'work_year'],
        title=f"Salary Distribution {f'for {viz_job_title}' if viz_job_title != 'All' else '(All Jobs)'}",
        labels={'salary_in_usd': 'Salary (USD)', 'work_year_jitter': 'Year', 'experience_level': 'Experience'},
        opacity=0.6
    )
    fig.update_xaxes(tickmode='linear', tick0=2020, dtick=1)
    fig.update_layout(height=600)
    
else:  # Salary Trends
    # Line chart showing average trends
    trend_data = df_viz.groupby(['work_year', 'experience_level'])['salary_in_usd'].mean().reset_index()
    
    fig = px.line(
        trend_data,
        x='work_year',
        y='salary_in_usd',
        color='experience_level',
        markers=True,
        title=f"Average Salary Trends by Experience Level {f'for {viz_job_title}' if viz_job_title != 'All' else '(All Jobs)'}",
        labels={'salary_in_usd': 'Average Salary (USD)', 'work_year': 'Year', 'experience_level': 'Experience'}
    )
    fig.update_layout(height=600)

# Add your prediction as a highlighted point (for scatter and trends only)
if viz_type in ["Scatter Plot", "Salary Trends"]:
    fig.add_scatter(
        x=[calc_work_year], 
        y=[calc_predicted_salary], 
        mode='markers', 
        marker=dict(color='red', size=20, symbol='star', line=dict(color='white', width=2)), 
        name='Your Prediction',
        hovertemplate=f'<b>Your Prediction</b><br>Year: {calc_work_year}<br>Salary: ${calc_predicted_salary:,.0f}<extra></extra>'
    )

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Additional Statistics
# ---------------------------------------------------------
st.markdown("---")
st.header("📈 Salary Statistics by Job Title")

# Get top 10 paying jobs
top_jobs = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10).reset_index()

fig_bar = px.bar(
    top_jobs,
    x='salary_in_usd',
    y='job_title',
    orientation='h',
    title='Top 10 Highest Paying Job Titles',
    labels={'salary_in_usd': 'Average Salary (USD)', 'job_title': 'Job Title'},
    color='salary_in_usd',
    color_continuous_scale='Viridis'
)
fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=500)
st.plotly_chart(fig_bar, use_container_width=True)

# Summary statistics
st.markdown("---")
st.subheader("📊 Dataset Summary")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric("Total Records", f"{len(df):,}")
with stat_col2:
    st.metric("Average Salary", f"${df['salary_in_usd'].mean():,.0f}")
with stat_col3:
    st.metric("Median Salary", f"${df['salary_in_usd'].median():,.0f}")
with stat_col4:
    st.metric("Unique Job Titles", f"{df['job_title'].nunique()}")

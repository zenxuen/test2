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
st.header("📊 Interactive Data Visualization")

# Create tabs for different visualizations
viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["📈 Salary Trends", "🎯 Interactive Scatter", "📊 Distribution", "🔍 Job Comparison"])

# Sidebar filters
st.sidebar.header("🎨 Visualization Filters")
viz_job_filter = st.sidebar.multiselect(
    "Filter by Job Titles", 
    options=sorted(df['job_title'].unique()),
    default=[]
)
viz_exp_filter = st.sidebar.multiselect(
    "Filter by Experience Level",
    options=sorted(df['experience_level'].unique()),
    default=[]
)

# Apply filters
df_filtered = df.copy()
if viz_job_filter:
    df_filtered = df_filtered[df_filtered['job_title'].isin(viz_job_filter)]
if viz_exp_filter:
    df_filtered = df_filtered[df_filtered['experience_level'].isin(viz_exp_filter)]

# Tab 1: Animated Line Chart
with viz_tab1:
    st.subheader("💫 Salary Trends Over Time")
    
    # Group by year and experience level
    trend_data = df_filtered.groupby(['work_year', 'experience_level'])['salary_in_usd'].agg(['mean', 'count']).reset_index()
    trend_data.columns = ['work_year', 'experience_level', 'avg_salary', 'count']
    
    fig_trend = px.line(
        trend_data,
        x='work_year',
        y='avg_salary',
        color='experience_level',
        markers=True,
        title='Average Salary Trends by Experience Level',
        labels={'avg_salary': 'Average Salary (USD)', 'work_year': 'Year', 'experience_level': 'Experience'},
        hover_data={'count': True}
    )
    
    # Add your prediction
    fig_trend.add_scatter(
        x=[calc_work_year], 
        y=[calc_predicted_salary], 
        mode='markers', 
        marker=dict(color='red', size=20, symbol='star', line=dict(color='white', width=3)), 
        name='Your Prediction',
        hovertemplate=f'<b>Your Prediction</b><br>Year: {calc_work_year}<br>Salary: ${calc_predicted_salary:,.0f}<extra></extra>'
    )
    
    fig_trend.update_layout(
        height=500,
        hovermode='x unified',
        xaxis=dict(dtick=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Additional stats
    col1, col2, col3 = st.columns(3)
    with col1:
        growth_2020_2022 = ((df_filtered[df_filtered['work_year']==2022]['salary_in_usd'].mean() - 
                             df_filtered[df_filtered['work_year']==2020]['salary_in_usd'].mean()) / 
                             df_filtered[df_filtered['work_year']==2020]['salary_in_usd'].mean() * 100) if len(df_filtered[df_filtered['work_year']==2020]) > 0 else 0
        st.metric("📈 Growth (2020-2022)", f"{growth_2020_2022:.1f}%")
    with col2:
        st.metric("📊 Records Shown", f"{len(df_filtered):,}")
    with col3:
        st.metric("💰 Avg Salary", f"${df_filtered['salary_in_usd'].mean():,.0f}")

# Tab 2: Interactive Scatter with Animation
with viz_tab2:
    st.subheader("🎯 Interactive Salary Explorer")
    
    # Add jitter for better visibility
    import numpy as np
    df_scatter = df_filtered.copy()
    df_scatter['work_year_jitter'] = df_scatter['work_year'] + np.random.uniform(-0.15, 0.15, len(df_scatter))
    
    # Color by selector
    color_by = st.selectbox("Color by:", ['experience_level', 'company_size', 'job_title'], key='color_scatter')
    size_by = st.selectbox("Size by:", ['remote_ratio', 'salary_in_usd'], key='size_scatter')
    
    fig_scatter = px.scatter(
        df_scatter,
        x='work_year_jitter',
        y='salary_in_usd',
        color=color_by,
        size=size_by,
        hover_data=['job_title', 'experience_level', 'company_size', 'remote_ratio', 'work_year'],
        title='Salary Distribution with Custom Dimensions',
        labels={'salary_in_usd': 'Salary (USD)', 'work_year_jitter': 'Year'},
        opacity=0.7
    )
    
    # Add your prediction
    fig_scatter.add_scatter(
        x=[calc_work_year], 
        y=[calc_predicted_salary], 
        mode='markers', 
        marker=dict(color='red', size=25, symbol='star', line=dict(color='white', width=3)), 
        name='Your Prediction',
        hovertemplate=f'<b>Your Prediction</b><br>Year: {calc_work_year}<br>Salary: ${calc_predicted_salary:,.0f}<extra></extra>'
    )
    
    fig_scatter.update_xaxes(tickmode='linear', tick0=2020, dtick=1)
    fig_scatter.update_layout(height=500, hovermode='closest')
    st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 3: Box Plot Distribution
with viz_tab3:
    st.subheader("📊 Salary Distribution Analysis")
    
    dist_view = st.radio("View by:", ['Year', 'Experience Level', 'Company Size'], horizontal=True)
    
    if dist_view == 'Year':
        fig_box = px.box(
            df_filtered,
            x='work_year',
            y='salary_in_usd',
            color='experience_level',
            title='Salary Distribution by Year and Experience',
            labels={'salary_in_usd': 'Salary (USD)', 'work_year': 'Year'}
        )
    elif dist_view == 'Experience Level':
        fig_box = px.box(
            df_filtered,
            x='experience_level',
            y='salary_in_usd',
            color='experience_level',
            title='Salary Distribution by Experience Level',
            labels={'salary_in_usd': 'Salary (USD)', 'experience_level': 'Experience'}
        )
    else:
        fig_box = px.box(
            df_filtered,
            x='company_size',
            y='salary_in_usd',
            color='company_size',
            title='Salary Distribution by Company Size',
            labels={'salary_in_usd': 'Salary (USD)', 'company_size': 'Company Size'}
        )
    
    fig_box.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Violin plot option
    show_violin = st.checkbox("Show Violin Plot (shows density)", value=False)
    if show_violin:
        fig_violin = px.violin(
            df_filtered,
            x='experience_level',
            y='salary_in_usd',
            color='experience_level',
            box=True,
            title='Salary Density Distribution by Experience Level',
            labels={'salary_in_usd': 'Salary (USD)', 'experience_level': 'Experience'}
        )
        fig_violin.update_layout(height=500)
        st.plotly_chart(fig_violin, use_container_width=True)

# Tab 4: Job Comparison
with viz_tab4:
    st.subheader("🔍 Compare Job Titles")
    
    # Select jobs to compare
    compare_jobs = st.multiselect(
        "Select up to 5 jobs to compare:",
        options=sorted(df['job_title'].unique()),
        default=sorted(df.groupby('job_title')['salary_in_usd'].mean().nlargest(3).index.tolist())[:3],
        max_selections=5
    )
    
    if compare_jobs:
        df_compare = df[df['job_title'].isin(compare_jobs)]
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Average salary comparison
            avg_by_job = df_compare.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=True).reset_index()
            fig_compare = px.bar(
                avg_by_job,
                x='salary_in_usd',
                y='job_title',
                orientation='h',
                title='Average Salary Comparison',
                labels={'salary_in_usd': 'Average Salary (USD)', 'job_title': 'Job Title'},
                color='salary_in_usd',
                color_continuous_scale='Viridis'
            )
            fig_compare.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_compare, use_container_width=True)
        
        with col2:
            # Salary over time
            job_trends = df_compare.groupby(['work_year', 'job_title'])['salary_in_usd'].mean().reset_index()
            fig_job_trend = px.line(
                job_trends,
                x='work_year',
                y='salary_in_usd',
                color='job_title',
                markers=True,
                title='Salary Trends Over Time',
                labels={'salary_in_usd': 'Average Salary (USD)', 'work_year': 'Year'}
            )
            fig_job_trend.update_layout(height=400, xaxis=dict(dtick=1))
            st.plotly_chart(fig_job_trend, use_container_width=True)
        
        # Statistical comparison table
        st.markdown("### 📊 Statistical Comparison")
        stats_df = df_compare.groupby('job_title')['salary_in_usd'].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std Dev', 'std')
        ]).round(0)
        
        st.dataframe(stats_df.style.format('${:,.0f}', subset=['Mean', 'Median', 'Min', 'Max', 'Std Dev']), use_container_width=True)
    else:
        st.info("👆 Select job titles above to compare")


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

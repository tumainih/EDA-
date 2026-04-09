import pandas as pd
import numpy as np
import plotly.express as px

from scipy.stats import skew
import plotly.graph_objects as go

def generate_descriptive_stats(df):
    """
    Returns descriptive statistics and detects skewness for numerical columns.
    """
    stats = df.describe(include='all').T
    num_cols = df.select_dtypes(include=[np.number]).columns
    if not num_cols.empty:
        stats['skewness'] = df[num_cols].apply(lambda x: skew(x.dropna()))
    return stats

def get_categorical_proportions(df):
    """
    Returns value counts and proportions for categorical columns.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    proportions = {}
    for col in cat_cols:
        counts = df[col].value_counts()
        props = df[col].value_counts(normalize=True)
        proportions[col] = pd.DataFrame({'Count': counts, 'Proportion': props})
    return proportions

def plot_correlation_matrix(df):
    """
    Generates an interactive correlation matrix plot.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    if numerical_df.shape[1] < 2:
        return None

    corr = numerical_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', 
                    title="Neural Feature Correlation Map", template="plotly_dark")
    fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color="#22d3ee"))
    return fig

def plot_feature_distributions(df, limit=6):
    """
    Plots interactive distributions for numerical features with skewness context.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    cols = numerical_df.columns[:limit]
    
    if len(cols) == 0:
        return None

    fig = px.histogram(numerical_df, x=cols, facet_col='variable', facet_col_wrap=3,
                       marginal='box', template='plotly_dark', barmode='overlay',
                       color_discrete_sequence=['#22d3ee', '#c084fc', '#f472b6'])

    fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color="#ffffff"), height=600)
    return fig

def plot_missing_matrix(df):
    """
    Visualizes missing data patterns.
    """
    missing = df.isnull()
    if not missing.any().any():
        return None
        
    fig = px.imshow(missing.astype(int), color_continuous_scale=['#0b1120', '#f472b6'],
                    labels=dict(x="Features", y="Samples", color="Missing"),
                    title="Intelligence Gap Analysis (Missing Data)", template="plotly_dark")
    fig.update_coloraxes(showscale=False)
    return fig

def compare_segments(df, segment_col, target_col):
    """
    Compares target variable across different segments.
    """
    if segment_col not in df.columns or target_col not in df.columns:
        return None
        
    if pd.api.types.is_numeric_dtype(df[target_col]):
        fig = px.box(df, x=segment_col, y=target_col, color=segment_col, 
                     points="all", title=f"Segment Variation: {target_col} by {segment_col}",
                     template="plotly_dark")
    else:
        fig = px.histogram(df, x=segment_col, color=target_col, barmode='group',
                           title=f"Segment Distribution: {target_col} by {segment_col}",
                           template="plotly_dark")
        
    fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120')
    return fig

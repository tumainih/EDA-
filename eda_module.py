import pandas as pd
import numpy as np
import plotly.express as px

def generate_descriptive_stats(df):
    """
    Returns descriptive statistics for numerical and categorical columns.
    """
    return df.describe(include='all')

def plot_correlation_matrix(df):
    """
    Generates an interactive correlation matrix plot using Plotly Express.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    if numerical_df.shape[1] < 2:
        return None

    corr = numerical_df.corr()
    
    # Futuristic neon correlation mapped interactively
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='curl', 
                    title="Numerical Feature Correlation Matrix", template="plotly_dark")
    
    # Adjust layout to fit neon vibes
    fig.update_layout(
        paper_bgcolor='#050505',
        plot_bgcolor='#050505',
        font=dict(color="#00f3ff"),
        title_font=dict(size=20, color="#9d00ff")
    )
    return fig

def plot_feature_distributions(df, limit=4):
    """
    Plots interactive distributions for up to `limit` numerical features using Plotly.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    cols = numerical_df.columns[:limit]
    
    if len(cols) == 0:
        return None

    # Melt DataFrame for Facet Grid plotting in Plotly
    melted_df = numerical_df[cols].melt(var_name='Feature', value_name='Value')

    fig = px.histogram(melted_df, x='Value', color='Feature', facet_row='Feature', 
                       marginal='box', template='plotly_dark', 
                       height=250 * len(cols), color_discrete_sequence=px.colors.qualitative.Set1)

    fig.update_layout(
        paper_bgcolor='#050505',
        plot_bgcolor='#050505',
        font=dict(color="#ffffff"),
        showlegend=False
    )
    return fig

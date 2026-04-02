import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(max_entries=1)
def clean_data(df):
    """
    Automatically cleans the dataframe:
    - Drops completely empty columns/rows
    - Drops duplicates
    - Imputes numerical missing values with median
    - Imputes categorical missing values with mode
    
    Cached via Streamlit to ensure scalable reprocessing skip on UI rerenders.
    """
    # Create a copy so we don't mutate unexpectedly
    cleaned_df = df.copy()

    # Drop empty columns and rows
    cleaned_df.dropna(how='all', axis=1, inplace=True)
    cleaned_df.dropna(how='all', axis=0, inplace=True)

    # Drop duplicates
    cleaned_df.drop_duplicates(inplace=True)

    # Missing value imputation
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # Fill numeric with median
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                # Fill categorical with mode
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
                else:
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")

    return cleaned_df

def get_data_summary(df, cleaned_df):
    """
    Returns a brief dictionary of cleaning operations performed.
    """
    original_rows, original_cols = df.shape
    new_rows, new_cols = cleaned_df.shape
    
    missing_before = df.isnull().sum().sum()
    missing_after = cleaned_df.isnull().sum().sum()
    
    duplicates_dropped = original_rows - new_rows
    
    return {
        "original_shape": f"{original_rows} rows, {original_cols} columns",
        "new_shape": f"{new_rows} rows, {new_cols} columns",
        "missing_values_handled": missing_before - missing_after,
        "duplicates_dropped": duplicates_dropped
    }

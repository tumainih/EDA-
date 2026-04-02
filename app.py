import streamlit as st
import pandas as pd
from data_cleaning import clean_data, get_data_summary
from eda_module import generate_descriptive_stats, plot_correlation_matrix, plot_feature_distributions
from ml_engine import train_model
from prescription import generate_recommendations

# Setup page config for a wider layout
st.set_page_config(page_title="AI Data Analytics", layout="wide", page_icon="⚡")

# Futuristic Header
st.markdown("<h1 style='color: #00f3ff; text-shadow: 0 0 10px rgba(0,243,255,0.4);'>AI-Powered Data Analytics Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8a8a93;'>Upload a structured dataset to automatically generate Insights, Explanations, Predictions, and Recommendations.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])

@st.cache_data(max_entries=1)
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if uploaded_file is not None:
    # 1. Read Data (Cached globally)
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    if df.empty:
        st.error("Uploaded dataset is empty.")
        st.stop()

    # 2. Automated Cleaning
    with st.spinner("Initiating Auto-Cleaning protocol..."):
        cleaned_df = clean_data(df)
        cleaning_summary = get_data_summary(df, cleaned_df)

    st.success(f"Data ingested and cleaned successfully. Transformed from {cleaning_summary['original_shape']} to {cleaning_summary['new_shape']}.")
    
    # [EXPORT] Output cleaned dataframe
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Export Cleaned Dataset (CSV)", data=csv, file_name="cleaned_data.csv", mime="text/csv")
    
    # 3. Tabulated One-Click Interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Insights", "🧠 Explanations", "🎯 Predictions", "💡 Recommendations", "🤖 Ops Assistant"])

    with tab1:
        st.markdown("<h3 style='color: #9d00ff;'>Descriptive Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(cleaned_df.describe(), use_container_width=True)
        
        st.markdown("<h3 style='color: #9d00ff;'>Feature Distributions</h3>", unsafe_allow_html=True)
        dist_fig = plot_feature_distributions(cleaned_df)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
        else:
            st.info("Not enough numerical columns to plot distributions.")

    with tab2:
        st.markdown("<h3 style='color: #00f3ff;'>Correlation Analysis</h3>", unsafe_allow_html=True)
        st.write("Variables grouped by how strongly they trend together.")
        corr_fig = plot_correlation_matrix(cleaned_df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Not enough numerical columns to calculate correlations.")

    with tab3:
        st.markdown("<h3 style='color: #ff00ff;'>Automated ML Modeling</h3>", unsafe_allow_html=True)
        target_col = st.selectbox("Select Target Variable for Prediction:", cleaned_df.columns)
        
        if st.button("Run Predictive Engine"):
            with st.spinner("Training Neural/Tree models..."):
                results = train_model(cleaned_df, target_col)
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.session_state['ml_results'] = results
                    st.session_state['target_col'] = target_col
            
            if 'ml_results' in st.session_state:
                res = st.session_state['ml_results']
                st.write(f"**Task Detected:** {res['task_type']}")
                st.metric(label=res['metric_name'], value=f"{res['score']:.4f}")
                
                st.write("**Top Feature Importances**")
                st.dataframe(res['feature_importances'], use_container_width=True)
                st.bar_chart(res['feature_importances'].set_index('Feature'))
                
                # [EXPORT] ML Importances Tracker
                csv_ml = res['feature_importances'].to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Export ML Importances (CSV)", data=csv_ml, file_name="ml_importances.csv", mime="text/csv")

    with tab4:
        st.markdown("<h3 style='color: #ffffff; text-shadow: 0 0 10px #ffffff;'>Actionable Recommendations</h3>", unsafe_allow_html=True)
        if 'ml_results' in st.session_state:
            recs = generate_recommendations(st.session_state['ml_results']['feature_importances'], st.session_state['ml_results']['task_type'])
            for r in recs:
                st.info(r)
                
            # [EXPORT] Recommendatons Tracker
            recs_text = "\n\n".join(recs)
            st.download_button(label="📥 Export Actionable Prescriptions (TXT)", data=recs_text, file_name="strategic_prescriptions.txt", mime="text/plain")
        else:
            st.warning("Please run the Predictive Engine in the 'Predictions' tab first to generate recommendations.")

    with tab5:
        st.markdown("<h3 style='color: #00f3ff;'>Operations Assistant</h3>", unsafe_allow_html=True)
        st.write("Need help understanding the platform operations, ML functionality, or Data Science terminology? Ask me!")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "bot", "content": "Hello! I am your embedded Operations Assistant. How can I help clarify your analysis today?"}]
            
        for msg in st.session_state.messages:
            st.chat_message("assistant" if msg["role"] == "bot" else "user").write(msg["content"])
            
        if prompt := st.chat_input("Ask a question about your analysis..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            p = prompt.lower()
            response = "I am the local Ops Bot! Ask me about 'EDA', 'Random Forest', 'Missing Values', or what 'Feature Importance' means!"
            if 'random forest' in p or 'ml' in p or 'predict' in p:
                response = "Our ML engine (Predictions tab) leverages Random Forest, which is a powerful ensemble of hundreds of decision trees! This means instead of failing if one feature acts weirdly, it intelligently votes on the best outcome."
            elif 'eda' in p or 'correlation' in p or 'explain' in p:
                response = "Exploratory Data Analysis (Explanations tab) helps reveal patterns. In our Correlation Matrix: when one variable goes up, if another goes up perfectly, it has a +1.0 score. If it moves oppositely, it scores -1.0. This explains relationships!"
            elif 'missing' in p or 'clean' in p:
                response = "Data isn't perfect! Our automated cleaning pipeline drops completely empty columns and duplicate rows. For missing math values, we safely fill them with the MEDIAN (middle value) so extreme outliers don't pollute the data."
            elif 'recommendation' in p or 'feature importance' in p:
                response = "Feature Importance mathematically dictates which variable in your dataset most strongly shifted your prediction curve. The Recommendations tab transforms these high-priority drivers into human-readable action-items."
                
            st.session_state.messages.append({"role": "bot", "content": response})
            st.chat_message("assistant").write(response)

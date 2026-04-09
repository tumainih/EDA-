import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import io
from data_cleaning import clean_data, get_data_summary
from eda_module import (
    generate_descriptive_stats, 
    get_categorical_proportions,
    plot_correlation_matrix, 
    plot_feature_distributions,
    plot_missing_matrix,
    compare_segments
)
from ml_engine import model_duel, get_feature_importance, predict_instance
from prescription import generate_recommendations, get_bot_response

# --- PRISM UI CONFIGURATION (ULTRA-VIBRANT) ---
st.set_page_config(page_title="dataMaster | Prism Intelligence", layout="wide", page_icon="🌈")

def safe_execute(func, *args, **kwargs):
    try: return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Operational Exception: {e}")
        return None

# Prism Shifting Background CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    * { font-family: 'Outfit', sans-serif; }
    
    .block-container { padding-top: 1.5rem !important; padding-bottom: 0rem !important; }
    
    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #312e81, #020617);
        background-size: 400% 400%;
        animation: prismGradient 15s ease infinite;
        color: #f8fafc;
    }
    
    @keyframes prismGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .brand-title {
        font-size: 3.2rem; font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6, #fbbf24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: -5px; filter: drop-shadow(0 0 12px rgba(244, 114, 182, 0.4));
    }
    .brand-sub { color: #94a3b8; letter-spacing: 6px; font-size: 0.6rem; margin-bottom: 1.5rem; font-weight: 600; opacity: 0.8; }
    
    /* Prism Bento Cards */
    .bento-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(25px);
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .bento-card:hover { 
        border: 1px solid rgba(244, 114, 182, 0.5); 
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(244, 114, 182, 0.15);
    }
    
    .insight-note {
        background: rgba(129, 140, 248, 0.1);
        border-left: 4px solid #f472b6;
        padding: 0.8rem; border-radius: 0 8px 8px 0;
        margin-top: 0.5rem; font-size: 0.85rem; color: #e2e8f0;
    }
    
    /* Variable-Color Hero Metrics */
    .metric-cyan { color: #22d3ee; text-shadow: 0 0 20px rgba(34, 211, 238, 0.5); }
    .metric-magenta { color: #f472b6; text-shadow: 0 0 20px rgba(244, 114, 182, 0.5); }
    .metric-gold { color: #fbbf24; text-shadow: 0 0 20px rgba(251, 191, 36, 0.5); }
    .metric-emerald { color: #34d399; text-shadow: 0 0 20px rgba(52, 211, 153, 0.5); }
    
    .metric-val { font-size: 2.5rem; font-weight: 800; }
    .badge {
        display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px;
        font-size: 0.65rem; font-weight: 800; background: rgba(255,255,255,0.05); color: #94a3b8;
        border: 1px solid rgba(255,255,255,0.1); margin-bottom: 0.5rem;
    }
    
    /* Prism Sidebar */
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #020617, #0f172a) !important; 
        border-right: 1px solid rgba(244, 114, 182, 0.1); 
    }
</style>
""", unsafe_allow_html=True)

# --- GLOBAL STATE ---
if 'data' not in st.session_state: st.session_state.data = None
if 'model_res' not in st.session_state: st.session_state.model_res = None
if 'chat_stack' not in st.session_state: st.session_state.chat_stack = []

with st.sidebar:
    st.markdown("<h1 style='color: #f472b6; margin-bottom: 0;'>dataMaster</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.5rem; color: #818cf8; font-weight: 800;'>PRISM EVOLUTION // AURORA 3.0</p>", unsafe_allow_html=True)
    st.divider()
    pillar = st.radio("Intelligence Pillar", ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])
    st.divider()
    up_file = st.file_uploader("Spectral Ingestion", type=['csv','xlsx','json','sav','dta','txt'])

@st.cache_data(show_spinner=False)
def load_stream(file):
    if file is None: return None
    name = file.name.lower()
    try:
        if name.endswith('.csv'): return pd.read_csv(file)
        elif name.endswith(('.xlsx', '.xls')): return pd.read_excel(file)
        elif name.endswith('.json'): return pd.read_json(file)
        elif name.endswith('.sav'): return pd.read_spss(file)
        elif name.endswith('.dta'): return pd.read_stata(file)
        else: return pd.read_csv(file, sep=None, engine='python')
    except Exception as e: return f"Error: {e}"

if up_file:
    raw_df = load_stream(up_file)
    if isinstance(raw_df, str): st.error(raw_df)
    elif raw_df is not None:
        if st.session_state.data is None or up_file.name != st.session_state.get('fid'):
            st.session_state.data = clean_data(raw_df)
            st.session_state.fid = up_file.name
        
        c_df = st.session_state.data
        q = get_data_summary(raw_df, c_df)
        
        # Prism Header
        st.markdown(f"<h1 class='brand-title'>{pillar}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='brand-sub'>PRISM ANALYTICS PROTOCOL // SYNC SECURE</p>", unsafe_allow_html=True)

        # --- PILLAR 1: DESCRIPTIVE ---
        if pillar == "Descriptive":
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='badge'>HEALTH INTEGRITY</div><div class='metric-val metric-cyan'>{int((1 - q['total_missing_before']/raw_df.size)*100)}%</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='badge'>DATA VOLUME (ROWS)</div><div class='metric-val metric-magenta'>{c_df.shape[0]}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='badge'>INFOSPHERE (CELLS)</div><div class='metric-val metric-gold'>{c_df.size}</div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='badge'>DUPLICATE FLUTTER</div><div class='metric-val metric-emerald'>{q['duplicates_found']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            cd1, cd2 = st.columns([1, 1.25])
            with cd1:
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                st.markdown("### 📋 Computational Signature")
                st.dataframe(generate_descriptive_stats(c_df), height=350, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with cd2:
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                st.markdown("### 📈 Spectral Distributions")
                fig = safe_execute(plot_feature_distributions, c_df, limit=4)
                if fig: 
                    fig.update_layout(colorway=['#22d3ee', '#f472b6', '#fbbf24', '#34d399'])
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown("<div class='insight-note'>Distribution patterns identified and color-indexed for spectral clarity.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # --- PILLAR 2: DIAGNOSTIC ---
        elif pillar == "Diagnostic":
            dc1, dc2 = st.columns([1, 1])
            with dc1:
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                st.markdown("### 🕸️ Prism Interaction Map")
                fig = safe_execute(plot_correlation_matrix, c_df)
                if fig: 
                    fig.update_layout(coloraxis_colorscale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("<div class='insight-note'>Vibrant areas represent strong feature coupling.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with dc2:
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                st.markdown("### 🔬 Spectral Gaps")
                fig = safe_execute(plot_missing_matrix, raw_df)
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.success("Intelligence stream is 100% complete.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            st.markdown("### 👥 Segment Prism")
            sc1, sc2 = st.columns(2)
            s = sc1.selectbox("Filter Segment", c_df.columns)
            t = sc2.selectbox("Performance Prism", [c for c in c_df.columns if c != s])
            fig = safe_execute(compare_segments, c_df, s, t)
            if fig: 
                fig.update_layout(colorway=['#f472b6', '#22d3ee', '#fbbf24', '#34d399'])
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- PILLAR 3: PREDICTIVE ---
        elif pillar == "Predictive":
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            pc1, pc2 = st.columns([1, 2])
            target = pc1.selectbox("Predictive Target", c_df.columns, index=len(c_df.columns)-1)
            feats = pc2.multiselect("Predictors", [c for c in c_df.columns if c != target], default=[c for c in c_df.columns if c != target][:8])
            if st.button("🚀 Fire Prism Duel"):
                with st.spinner("Competing for Spectral Dominance..."):
                    st.session_state.model_res = model_duel(c_df[feats + [target]], target)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.model_res:
                m = st.session_state.model_res
                mc1, mc2 = st.columns([1.25, 1])
                with mc1:
                    st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                    st.markdown(f"#### 🏆 Optimum Engine: {m['best_model_name']}")
                    st.table(m['comparison'].style.background_gradient(cmap='YlGnBu'))
                    st.markdown("</div>", unsafe_allow_html=True)
                with mc2:
                    st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                    st.markdown("#### 🦾 Prism What-If")
                    with st.form("prism_wi"):
                        wi_data = {}
                        wi_cols = st.columns(2)
                        for i, f in enumerate(feats):
                            with wi_cols[i % 2]:
                                if pd.api.types.is_numeric_dtype(c_df[f]):
                                    wi_data[f] = st.number_input(f, value=float(c_df[f].median()))
                                else: wi_data[f] = st.selectbox(f, c_df[f].unique())
                        if st.form_submit_button("Generate Predictive Prism"):
                            inst = pd.DataFrame([wi_data])
                            for c in inst.select_dtypes(include=['object']).columns:
                                inst[c] = pd.Categorical(inst[c], categories=c_df[c].unique()).codes
                            p = predict_instance(m['models'][m['best_model_name']], inst)
                            st.markdown(f"<div class='metric-val metric-magenta'>{p[0]}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        # --- PILLAR 4: PRESCRIPTIVE ---
        elif pillar == "Prescriptive":
            if not st.session_state.model_res: st.warning("Requires Predictive Duel Activation.")
            else:
                m = st.session_state.model_res
                imp = get_feature_importance(m['models'][m['best_model_name']], m['X'].columns)
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                st.markdown("### 📜 Prism Strategic Protocols")
                st.write(generate_recommendations(imp, "reg" if "MAE" in m['comparison'].columns else "clf"))
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                st.markdown("### 🗨️ Prism Intelligence Bot")
                for msg in st.session_state.chat_stack:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                if pr := st.chat_input("Query Prism..."):
                    st.session_state.chat_stack.append({"role": "user", "content": pr})
                    with st.chat_message("user"): st.markdown(pr)
                    ctx = {"top_feature": imp.iloc[0]['Feature'], "missing_total": q['total_missing_before']}
                    ans = get_bot_response(pr, ctx)
                    with st.chat_message("assistant"): st.markdown(ans)
                    st.session_state.chat_stack.append({"role": "assistant", "content": ans})
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<h1 class='brand-title'>dataMaster</h1>", unsafe_allow_html=True)
    st.markdown("<p class='brand-sub'>PRISM ANALYTICS ENGINE READY // AWAITING STREAM</p>", unsafe_allow_html=True)
    st.markdown("<div class='bento-card' style='text-align: center; border: 2px dashed rgba(129, 140, 248, 0.4);'>", unsafe_allow_html=True)
    st.markdown("### ⚡ Ingest Ingest Stream (CSV, EXCEL, SAV, DTA)")
    st.write("dataMaster will automatically harmonize and color-index your intelligence.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.sidebar.markdown("<div style='position: fixed; bottom: 5px; opacity: 0.3; font-size: 0.5rem; color: #818cf8;'>PRISM EVOLUTION // STABLE CORE</div>", unsafe_allow_html=True)

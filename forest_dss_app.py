import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os

st.set_page_config(page_title="Forest DSS", layout="wide", page_icon="🌳")

st.markdown("""
<style>
    .main > .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3 { margin-top: 0 !important; }
    .stApp { background-color: #f8f9fa; }
    .card {
        background: white; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center; border-left: 4px solid #2e7d32;
    }
    .metric-card .label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; margin: 4px 0; }
    .metric-card .delta { font-size: 0.8rem; color: #2e7d32; }
    .sidebar-header {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center;
    }
    .sidebar-header h3 { margin: 0; font-size: 1rem; color: white !important; }
    .sidebar-header p { margin: 4px 0 0; font-size: 0.75rem; opacity: 0.85; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.3px;
    }
    .badge-green { background: #e8f5e9; color: #2e7d32; }
    .badge-red { background: #ffebee; color: #c62828; }
    .badge-yellow { background: #fff8e1; color: #f57f17; }
    hr { margin: 1rem 0; opacity: 0.2; }
    .stDataFrame { border: 1px solid #e9ecef; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-weight: 700; }
    .stButton button {
        border-radius: 8px; font-weight: 600; padding: 0.4rem 1.2rem;
        transition: all 0.2s;
    }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    div[data-testid="stMetric"] {
        background: white; padding: 0.8rem 1rem; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-left: 3px solid #2e7d32;
    }
    .st-emotion-cache-1wivap2 { background-color: transparent !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'forest_inventory_dataset_1000 - forest_inventory_dataset_1000.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <div style="font-size:2rem;">🌳</div>
    <div>
        <h1 style="margin:0; font-size:1.6rem; color:#1b5e20;">DENR Forest Decision Support System</h1>
        <p style="margin:0; color:#6c757d; font-size:0.85rem;">
            AI-Powered Analytics for Sustainable Forest Management
        </p>
    </div>
</div>
<hr style="margin:0.8rem 0;">
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("""
<div class="sidebar-header">
    <h3>🌳 Navigation</h3>
    <p>Forest DSS v1.0</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "📊 Data Overview",
    "🌱 Survival Prediction",
    "⚠️ Mortality Risk",
    "🌿 Species Recommendation",
    "📈 Growth Prediction",
    "🌳 Carbon Storage",
    "📦 Timber Volume",
    "🗺️ GIS Priority Mapping",
    "📰 News Report Summary",
    "📋 Reforestation Project Monitoring Dataset",
], label_visibility="collapsed")

# ── Helpers ──
@st.cache_data
def train_survival_model():
    df_m = df.drop('Tree_ID', axis=1).copy()
    cat_cols = ['Species', 'Barangay', 'Municipality', 'Soil_Type']
    enc = {}
    for c in cat_cols:
        le = LabelEncoder()
        df_m[c] = le.fit_transform(df_m[c])
        enc[c] = le
    df_m['Survival_Status'] = LabelEncoder().fit_transform(df_m['Survival_Status'])
    X = df_m.drop('Survival_Status', axis=1)
    y = df_m['Survival_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf, enc, X.columns.tolist(), X_test, y_test

rf_model, encoders, feature_cols, X_test, y_test = train_survival_model()

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 11
})

# ── Ollama AI Assistant ──
OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_ollama(prompt, model="phi3:latest"):
    try:
        r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 800}}, timeout=60)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except:
        pass
    return ""

@st.cache_data(ttl=300)
def cached_ollama(prompt, model):
    return ask_ollama(prompt, model)

AI_MODELS = ["phi3:latest", "qwen3:4b", "tinyllama:latest", "deepseek-r1:8b", "qwen2.5-coder:7b"]
AI_REGION_CONTEXT = (
    "You are a forestry AI expert for DENR Caraga, Region XIII, Philippines. "
    "Limit all analysis, assumptions, examples, and recommendations to the Caraga Region only, "
    "including Agusan del Norte, Agusan del Sur, Surigao del Norte, Surigao del Sur, "
    "and Dinagat Islands. Do not generalize to other Philippine regions unless explicitly asked."
)

# ── Sidebar AI Section ──
st.sidebar.markdown("---")
st.sidebar.markdown("""<div style="background:#e8f5e9; padding:0.6rem 0.8rem; border-radius:8px; margin-bottom:0.5rem; border-left:3px solid #2e7d32;">
    <div style="font-size:0.8rem; font-weight:700; color:#1b5e20;">🤖 AI Assistant</div></div>""", unsafe_allow_html=True)
ai_model = st.sidebar.selectbox("Model", AI_MODELS, index=0, key="ai_model", label_visibility="collapsed")
ai_question = st.sidebar.text_input("Ask anything about the forest data...", placeholder="e.g. What actions improve survival?")
if ai_question:
    with st.spinner("Thinking..."):
        ans = cached_ollama(f"{AI_REGION_CONTEXT}\n\nAnswer concisely: {ai_question}", ai_model)
    if ans:
        st.sidebar.info(ans)
    else:
        st.sidebar.warning("AI unavailable. Check Ollama.")

# ── Page Context for AI ──
ai_context = {
    "total_trees": len(df),
    "species_count": df['Species'].nunique(),
    "municipalities": df['Municipality'].nunique(),
    "survival_rate": f"{(df['Survival_Status'] == 'Alive').mean():.1%}",
    "alive": int((df['Survival_Status'] == 'Alive').sum()),
    "dead": int((df['Survival_Status'] == 'Dead').sum()),
}

def recommendation_gauge(label, value, sublabel="", color="#2e7d32"):
    fig, ax = plt.subplots(figsize=(6.5, 0.9))
    ax.barh([0], [1], color='#e9ecef', height=0.4, edgecolor='none')
    ax.barh([0], [min(value, 1)], color=color, height=0.4, edgecolor='none')
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6)
    ax.text(min(value, 1) / 2, 0, f"{min(value, 1):.0%}", ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    ax.text(1.02, 0, label, va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
    ax.axis('off')
    if sublabel:
        ax.text(0.35, -0.4, sublabel, fontsize=8, color='#666', va='top', ha='center')
    plt.close(fig)
    return fig

def show_ai_analysis(prompt, key_suffix="", gauge_label=None, gauge_value=None, gauge_color="#2e7d32"):
    full_prompt = AI_REGION_CONTEXT + "\n\n" + prompt + (
        "\n\nConclude your response with '**Therefore, I recommend:**' followed by a clear ACTIONABLE RECOMMENDATION section with:"
        "\n1. NEXT STEPS — list the specific tasks that should be done immediately, in order."
        "\n2. MATERIALS NEEDED — list the tools, equipment, supplies, or resources required."
        "\n3. WHO SHOULD ACT — specify which DENR office, personnel, or stakeholder should take action."
        "\n4. TIMEFRAME — urgent (within days), short-term (weeks), or long-term (months)."
        "\nBe specific and practical for DENR Caraga field operations."
    )
    with st.expander("🤖 AI Analysis & Recommendations", expanded=False):
        if gauge_label is not None and gauge_value is not None:
            fig = recommendation_gauge(gauge_label, gauge_value, color=gauge_color)
            st.pyplot(fig, use_container_width=True)
        with st.spinner("Analyzing with AI..."):
            resp = cached_ollama(full_prompt, ai_model)
        if resp:
            st.markdown(resp)
        else:
            st.info("AI analysis unavailable. Ensure Ollama is running.")

# ── 1. DATA OVERVIEW ──
if page == "📊 Data Overview":
    alive_count = (df['Survival_Status'] == 'Alive').sum()
    dead_count = (df['Survival_Status'] == 'Dead').sum()
    surv_rate = alive_count / len(df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")
    cols = st.columns(5)
    cols[0].metric("Total Trees", df.shape[0])
    cols[1].metric("Species", df['Species'].nunique())
    cols[2].metric("Municipalities", df['Municipality'].nunique())
    cols[3].metric("Alive", f"{alive_count} ({surv_rate:.1%})")
    cols[4].metric("Dead", f"{dead_count} ({1-surv_rate:.1%})")
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Survival by Species")
        with st.spinner("Rendering chart..."):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            c = sns.countplot(data=df, y='Species', hue='Survival_Status', ax=ax,
                              palette={'Alive': '#2e7d32', 'Dead': '#c62828'})
            ax.set_xlabel('Count'); ax.set_ylabel('')
            ax.legend(loc='lower right')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Survival by Soil Type")
        with st.spinner("Rendering chart..."):
            fig, ax = plt.subplots(figsize=(9, 4))
            c = sns.countplot(data=df, x='Soil_Type', hue='Survival_Status', ax=ax,
                              palette={'Alive': '#2e7d32', 'Dead': '#c62828'})
            ax.set_xlabel(''); ax.set_ylabel('Count')
            ax.legend(loc='upper right')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("View Raw Data & Descriptive Statistics"):
        tab1, tab2 = st.tabs(["Data Sample", "Statistics"])
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
        with tab2:
            st.dataframe(df.describe(), use_container_width=True)

    show_ai_analysis(f"Analyze this Caraga Region forest inventory dataset: {ai_context['total_trees']} trees, {ai_context['species_count']} species, across {ai_context['municipalities']} municipalities. Survival rate: {ai_context['survival_rate']} ({ai_context['alive']} alive, {ai_context['dead']} dead). Explain key patterns in survival by species and soil type, and give management recommendations for DENR Caraga.", "data",
        gauge_label="Overall Survival Rate", gauge_value=surv_rate, gauge_color="#2e7d32" if surv_rate >= 0.7 else "#f57f17" if surv_rate >= 0.5 else "#c62828")

# ── 2. SURVIVAL PREDICTION ──
elif page == "🌱 Survival Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌱 Tree Survival Prediction")
    st.markdown("Enter tree details to predict whether the tree will survive.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        species = st.selectbox("Species", df['Species'].unique())
        barangay = st.selectbox("Barangay", df['Barangay'].unique())
        municipality = st.selectbox("Municipality", df['Municipality'].unique())
        soil = st.selectbox("Soil Type", df['Soil_Type'].unique())
    with col2:
        age = st.slider("Age (Years)", 1, 50, 10)
        height = st.slider("Height (m)", 0.5, 40.0, 15.0)
        diameter = st.slider("Diameter (cm)", 1.0, 60.0, 20.0)
        lat = st.number_input("Latitude", value=8.5, format="%.5f")
        lng = st.number_input("Longitude", value=126.0, format="%.5f")

    if st.button("Predict Survival", type="primary"):
        with st.spinner("Computing prediction..."):
            inp = pd.DataFrame([[species, barangay, municipality, lat, lng, age, height, diameter, soil]],
                               columns=['Species', 'Barangay', 'Municipality', 'Latitude', 'Longitude',
                                        'Age_Years', 'Height_m', 'Diameter_cm', 'Soil_Type'])
            for c in ['Species', 'Barangay', 'Municipality', 'Soil_Type']:
                inp[c] = encoders[c].transform(inp[c])
            prob = rf_model.predict_proba(inp)[0, 1]
            pred = rf_model.predict(inp)[0]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-card" style="border-left-color: {'#2e7d32' if pred else '#c62828'}">
            <div class="label">Prediction</div>
            <div class="value">{'✅ Alive' if pred else '❌ Dead'}</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card" style="border-left-color: #1976d2">
            <div class="label">Survival Probability</div>
            <div class="value">{prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        c3.markdown(f"""
        <div class="metric-card" style="border-left-color: {'#2e7d32' if prob > 0.5 else '#c62828'}">
            <div class="label">Confidence</div>
            <div class="value">{'High' if prob > 0.75 else 'Medium' if prob > 0.5 else 'Low'}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob))

        show_ai_analysis(f"A {species} tree ({age} yrs, {height:.1f}m tall, {diameter:.1f}cm diameter) in {barangay}, {municipality}, Caraga Region, on {soil} soil was predicted {'Alive' if pred else 'Dead'} with {prob:.1%} probability. Explain the likely factors influencing this prediction and recommend management actions for DENR Caraga.", "survival",
            gauge_label="Survival Probability", gauge_value=prob, gauge_color="#2e7d32" if prob >= 0.7 else "#f57f17" if prob >= 0.5 else "#c62828")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Model Performance")
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("ROC AUC", f"{auc:.3f}")
    st.text(classification_report(y_test, y_pred, target_names=['Dead', 'Alive']))
    st.markdown('</div>', unsafe_allow_html=True)

# ── 3. MORTALITY RISK ──
elif page == "⚠️ Mortality Risk":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚠️ Tree Mortality Risk Classification")
    st.markdown("Classifies trees into **Low**, **Medium**, or **High** mortality risk based on characteristics.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        species = st.selectbox("Species", df['Species'].unique(), key='mr_species')
        soil = st.selectbox("Soil Type", df['Soil_Type'].unique(), key='mr_soil')
        age = st.slider("Age (Years)", 1, 50, 10, key='mr_age')
    with col2:
        height = st.slider("Height (m)", 0.5, 40.0, 15.0, key='mr_ht')
        diameter = st.slider("Diameter (cm)", 1.0, 60.0, 20.0, key='mr_dia')
        municipality = st.selectbox("Municipality", df['Municipality'].unique(), key='mr_mun')

    if st.button("Classify Risk", type="primary", key='mr_btn'):
        with st.spinner("Classifying risk..."):
            inp = pd.DataFrame([[species, 'Libertad', municipality, 8.5, 126.0, age, height, diameter, soil]],
                               columns=['Species', 'Barangay', 'Municipality', 'Latitude', 'Longitude',
                                        'Age_Years', 'Height_m', 'Diameter_cm', 'Soil_Type'])
            for c in ['Species', 'Barangay', 'Municipality', 'Soil_Type']:
                inp[c] = encoders[c].transform(inp[c])
            prob_dead = rf_model.predict_proba(inp)[0, 0]

        color, badge, label, desc = ("#2e7d32", "🟢 Low",
            "Low Mortality Risk",
            "Tree is likely to survive. No immediate intervention needed.") if prob_dead < 0.25 else \
            ("#f57f17", "🟡 Medium",
            "Moderate Mortality Risk",
            "Tree faces moderate mortality risk. Monitor regularly.") if prob_dead < 0.50 else \
            ("#c62828", "🔴 High",
            "High Mortality Risk",
            "Tree is at high risk of mortality. Consider intervention.")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-card" style="border-left-color: {color}">
            <div class="label">Mortality Risk</div>
            <div class="value">{badge} {label}</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card" style="border-left-color: {color}">
            <div class="label">Death Probability</div>
            <div class="value">{prob_dead:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        c3.markdown(f"""
        <div class="metric-card" style="border-left-color: {color}">
            <div class="label">Survival Probability</div>
            <div class="value">{1-prob_dead:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        if prob_dead < 0.25:
            st.success(desc)
        elif prob_dead < 0.50:
            st.warning(desc)
        else:
            st.error(desc)

        show_ai_analysis(f"A {species} tree ({age} yrs, {height:.1f}m tall, {diameter:.1f}cm diameter) in {municipality}, Caraga Region, on {soil} soil has {label} mortality risk ({prob_dead:.1f}% death probability). Explain the risk factors and recommend specific interventions for DENR Caraga.", "mortality",
            gauge_label="Safety Score (1 - Death Risk)", gauge_value=1-prob_dead, gauge_color="#2e7d32" if prob_dead < 0.25 else "#f57f17" if prob_dead < 0.5 else "#c62828")

# ── 4. SPECIES RECOMMENDATION ──
elif page == "🌿 Species Recommendation":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌿 Species Recommendation Engine")
    st.markdown("Recommends the best tree species based on soil type and location.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        soil_type = st.selectbox("Soil Type", df['Soil_Type'].unique(), key='rec_soil')
    with col2:
        municipality = st.selectbox("Municipality", df['Municipality'].unique(), key='rec_mun')

    if st.button("Recommend Species", type="primary"):
        with st.spinner("Analyzing species data..."):
            sub = df[(df['Soil_Type'] == soil_type) & (df['Municipality'] == municipality)]
            if sub.empty:
                st.warning("No data for this combination. Showing best performers for this soil type.")
                sub = df[df['Soil_Type'] == soil_type]

            stats = sub.groupby('Species').agg(
                Count=('Tree_ID', 'count'),
                Survival_Rate=('Survival_Status', lambda x: (x == 'Alive').mean()),
                Avg_Height=('Height_m', 'mean'),
                Avg_Diameter=('Diameter_cm', 'mean')
            ).sort_values('Survival_Rate', ascending=False)

            best = stats.index[0]

        st.success(f"**Recommended Species: {best}** — {stats.loc[best, 'Survival_Rate']:.1%} survival rate in {soil_type} soil.")

        st.dataframe(stats.style.format({
            'Survival_Rate': '{:.1%}',
            'Avg_Height': '{:.1f} m',
            'Avg_Diameter': '{:.1f} cm'
        }), use_container_width=True)

        with st.spinner("Rendering chart..."):
            fig, ax = plt.subplots(figsize=(9, 4))
            colors = ['#2e7d32' if i == 0 else '#adb5bd' for i in range(len(stats))]
            sns.barplot(data=stats.reset_index(), x='Survival_Rate', y='Species',
                        palette=colors, ax=ax)
            ax.set_title(f'Species Survival Rate — {soil_type} Soil', fontweight='bold')
            ax.set_xlabel('Survival Rate')
            ax.set_ylabel('')
            st.pyplot(fig)

        show_ai_analysis(f"For {soil_type} soil in {municipality}, Caraga Region, the top recommended species is {best} with {stats.loc[best, 'Survival_Rate']:.1%} survival rate. Explain why this species thrives in these conditions and give reforestation advice for DENR Caraga.", "species_rec",
            gauge_label=f"Best Species: {best}", gauge_value=stats.loc[best, 'Survival_Rate'], gauge_color="#2e7d32")

# ── 5. GROWTH PREDICTION ──
elif page == "📈 Growth Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Tree Growth Prediction")
    st.markdown("Predict future height and diameter based on current age, species, and soil type.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        species = st.selectbox("Species", df['Species'].unique(), key='gr_species')
    with col2:
        soil = st.selectbox("Soil Type", df['Soil_Type'].unique(), key='gr_soil')
    with col3:
        current_age = st.slider("Current Age (Years)", 1, 40, 5, key='gr_age')
    future_age = st.slider("Predict at Age (Years)", current_age + 1, 50, current_age + 5, key='gr_fage')

    if st.button("Predict Growth", type="primary"):
        with st.spinner("Training growth model..."):
            sub = df[(df['Species'] == species) & (df['Soil_Type'] == soil)]
            if len(sub) < 5:
                sub = df[df['Species'] == species]
                st.info("Limited data for this specific combination; using species-level data.")

            sub = sub.copy()
            le_g = LabelEncoder()
            sub['Soil_Type_E'] = le_g.fit_transform(sub['Soil_Type'])

            Xg = sub[['Age_Years', 'Soil_Type_E']]
            rf_h = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_d = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_h.fit(Xg, sub['Height_m'])
            rf_d.fit(Xg, sub['Diameter_cm'])

            inp_g = pd.DataFrame([[future_age, le_g.transform([soil])[0]]], columns=['Age_Years', 'Soil_Type_E'])
            inp_c = pd.DataFrame([[current_age, le_g.transform([soil])[0]]], columns=['Age_Years', 'Soil_Type_E'])
            pred_h = rf_h.predict(inp_g)[0]
            pred_d = rf_d.predict(inp_g)[0]
            cur_h = rf_h.predict(inp_c)[0]
            cur_d = rf_d.predict(inp_c)[0]

        c1, c2 = st.columns(2)
        c1.markdown(f"""
        <div class="metric-card" style="border-left-color: #2e7d32">
            <div class="label">Height at Age {future_age}</div>
            <div class="value">{pred_h:.1f} m</div>
            <div class="delta">▲ {pred_h - cur_h:.1f} m from current ({cur_h:.1f} m)</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card" style="border-left-color: #5d4037">
            <div class="label">Diameter at Age {future_age}</div>
            <div class="value">{pred_d:.1f} cm</div>
            <div class="delta">▲ {pred_d - cur_d:.1f} cm from current ({cur_d:.1f} cm)</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Rendering growth chart..."):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            age_range = np.arange(1, 51)
            inp_range = pd.DataFrame({'Age_Years': age_range, 'Soil_Type_E': le_g.transform([soil])[0]})

            for ax, data, label, color, ylabel in [
                (axes[0], rf_h.predict(inp_range), 'Height', '#2e7d32', 'Height (m)'),
                (axes[1], rf_d.predict(inp_range), 'Diameter', '#5d4037', 'Diameter (cm)')
            ]:
                ax.plot(age_range, data, color=color, linewidth=2.5)
                ax.axvline(current_age, ls='--', color='gray', alpha=0.6, label=f'Current: {current_age} yrs')
                ax.axvline(future_age, ls='--', color='orange', alpha=0.6, label=f'Target: {future_age} yrs')
                ax.scatter([current_age], [data[current_age-1]], color='#1565c0', s=100, zorder=5, edgecolors='white')
                ax.scatter([future_age], [data[future_age-1]], color='orange', s=100, zorder=5, edgecolors='white')
                ax.set_xlabel('Age (Years)', fontweight='bold')
                ax.set_ylabel(ylabel, fontweight='bold')
                ax.set_title(f'{species} — {label} Growth Projection', fontweight='bold')
                ax.legend(frameon=True, fancybox=True)

            st.pyplot(fig)

        show_ai_analysis(f"A {species} tree on {soil} soil in the Caraga Region at age {current_age} is predicted to reach {pred_h:.1f}m height and {pred_d:.1f}cm diameter by age {future_age}. Explain the growth pattern and recommend optimal management timing for DENR Caraga.", "growth",
            gauge_label="Height Growth Rate", gauge_value=min((pred_h-cur_h)/cur_h, 1) if cur_h > 0 else 0, gauge_color="#2e7d32")

# ── 6. CARBON STORAGE ──
elif page == "🌳 Carbon Storage":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌳 Carbon Storage Estimation")
    st.markdown("Estimates above-ground carbon sequestration based on tree dimensions and species.")
    st.markdown('</div>', unsafe_allow_html=True)

    @st.cache_data
    def fit_carbon_model():
        sub = df.copy()
        sub['D2H'] = sub['Diameter_cm'] ** 2 * sub['Height_m']
        le_c = LabelEncoder()
        sub['Species_E'] = le_c.fit_transform(sub['Species'])
        Xc = sub[['D2H', 'Species_E']]
        yc = 0.05 * sub['D2H'] * 0.47
        rf_c = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_c.fit(Xc, yc)
        return rf_c, le_c

    rf_carb, carb_le = fit_carbon_model()

    col1, col2, col3 = st.columns(3)
    with col1:
        species = st.selectbox("Species", df['Species'].unique(), key='carb_sp')
    with col2:
        diameter = st.slider("Diameter at Breast Height (cm)", 5.0, 60.0, 25.0, key='carb_d')
    with col3:
        height = st.slider("Height (m)", 2.0, 40.0, 15.0, key='carb_h')

    if st.button("Estimate Carbon", type="primary"):
        with st.spinner("Computing carbon estimate..."):
            d2h_val = diameter ** 2 * height
            inp_c = pd.DataFrame([[d2h_val, carb_le.transform([species])[0]]], columns=['D2H', 'Species_E'])
            carbon_t = rf_carb.predict(inp_c)[0]
            co2_eq = carbon_t * 3.67

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-card" style="border-left-color: #2e7d32">
            <div class="label">Above-Ground Carbon</div>
            <div class="value">{carbon_t:.2f} t</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card" style="border-left-color: #1565c0">
            <div class="label">CO₂ Equivalent</div>
            <div class="value">{co2_eq:.2f} t</div>
        </div>
        """, unsafe_allow_html=True)
        c3.markdown(f"""
        <div class="metric-card" style="border-left-color: #5d4037">
            <div class="label">Total Biomass</div>
            <div class="value">{carbon_t / 0.47:.2f} t</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Methodology"):
            st.markdown("""
            **Allometric Equation:** Biomass = 0.05 × D² × H  
            **Carbon:** Biomass × 0.47  
            **CO₂e:** Carbon × 3.67  
            *D = diameter at breast height (cm), H = height (m)*
            """)

        show_ai_analysis(f"A {species} tree in the Caraga Region with {diameter}cm diameter and {height}m height stores approximately {carbon_t:.2f} tonnes of above-ground carbon ({co2_eq:.2f} tonnes CO₂e). Explain the importance of this species for carbon sequestration and climate change mitigation in Caraga.", "carbon",
            gauge_label="Carbon Sequestration Potential", gauge_value=min(carbon_t / 0.5, 1), gauge_color="#1565c0")

# ── 7. TIMBER VOLUME ──
elif page == "📦 Timber Volume":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📦 Timber Volume Prediction")
    st.markdown("Predicts merchantable timber volume using tree dimensions.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        species = st.selectbox("Species", df['Species'].unique(), key='tv_sp')
    with col2:
        diameter = st.slider("Diameter at Breast Height (cm)", 5.0, 60.0, 25.0, key='tv_d')
    with col3:
        height = st.slider("Height (m)", 2.0, 40.0, 15.0, key='tv_h')

    if st.button("Predict Volume", type="primary"):
        form_factor = 0.45
        volume_m3 = 0.00007854 * (diameter ** 2) * height * form_factor
        bd_ft = volume_m3 * 423.776

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-card" style="border-left-color: #5d4037">
            <div class="label">Volume</div>
            <div class="value">{volume_m3:.3f} m³</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card" style="border-left-color: #f57f17">
            <div class="label">Volume</div>
            <div class="value">{bd_ft:,.0f} bd ft</div>
        </div>
        """, unsafe_allow_html=True)
        c3.markdown(f"""
        <div class="metric-card" style="border-left-color: #1976d2">
            <div class="label">Form Factor</div>
            <div class="value">{form_factor}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Methodology"):
            st.markdown("""
            **Volume = 0.00007854 × D² × H × FF**  
            D = diameter (cm), H = height (m), FF = form factor (0.45)  
            1 m³ ≈ 423.78 board feet
            """)

        show_ai_analysis(f"A {species} tree in the Caraga Region with {diameter}cm DBH and {height}m height yields approximately {volume_m3:.3f} m³ ({bd_ft:,.0f} board feet) of timber. Explain the economic value and sustainable harvesting considerations for DENR Caraga.", "timber",
            gauge_label="Timber Volume Utilization", gauge_value=min(volume_m3 / 3.0, 1), gauge_color="#5d4037")

# ── 8. GIS PRIORITY MAPPING ──
elif page == "🗺️ GIS Priority Mapping":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺️ GIS-Based Reforestation Priority Mapping")
    st.markdown("Identifies high-priority areas for reforestation based on mortality rates and tree metrics.")
    st.markdown('</div>', unsafe_allow_html=True)

    agg = df.groupby(['Municipality', 'Barangay', 'Soil_Type']).agg(
        Tree_Count=('Tree_ID', 'count'),
        Mortality_Rate=('Survival_Status', lambda x: (x == 'Dead').mean()),
        Avg_Age=('Age_Years', 'mean'),
        Avg_Height=('Height_m', 'mean')
    ).reset_index()
    agg['Priority_Score'] = (
        agg['Mortality_Rate'] * 0.4 +
        (1 - agg['Avg_Height'] / df['Height_m'].max()) * 0.3 +
        (1 - agg['Avg_Age'] / df['Age_Years'].max()) * 0.3
    )
    agg['Priority'] = pd.cut(agg['Priority_Score'],
                             bins=[0, 0.33, 0.66, 1.0],
                             labels=['🟢 Low', '🟡 Medium', '🔴 High'])

    def color_priority(val):
        if '🔴' in str(val): return 'background-color: #ffebee'
        if '🟡' in str(val): return 'background-color: #fff8e1'
        if '🟢' in str(val): return 'background-color: #e8f5e9'
        return ''

    st.dataframe(
        agg.sort_values('Priority_Score', ascending=False).style.map(color_priority, subset=['Priority']),
        use_container_width=True
    )

    # ── Interactive Map ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺️ Tree Location Map")
    st.markdown("Click any tree marker to view details and open in Google Maps.")
    map_colors = st.radio("Color by", ["Survival Status", "Priority Score"], horizontal=True, key="map_mode")

    with st.spinner("Rendering interactive map..."):
        map_df = df[['Tree_ID', 'Latitude', 'Longitude', 'Survival_Status', 'Species', 'Barangay', 'Municipality', 'Age_Years', 'Height_m', 'Diameter_cm']].copy()
        avg_lat, avg_lon = df['Latitude'].mean(), df['Longitude'].mean()

        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10, tiles='CartoDB Positron', control_scale=True)

        if map_colors == "Survival Status":
            for _, row in map_df.iterrows():
                color = '#2e7d32' if row['Survival_Status'] == 'Alive' else '#c62828'
                status = '✅ Alive' if row['Survival_Status'] == 'Alive' else '❌ Dead'
                radius = max(3, row['Diameter_cm'] * 0.4)
                popup_html = f"""<div style="font-family:sans-serif;font-size:13px;min-width:200px;">
                    <b style="font-size:15px;">{row['Tree_ID']}</b><hr style="margin:4px 0;">
                    <b>Species:</b> {row['Species']}<br><b>Age:</b> {row['Age_Years']} yrs<br>
                    <b>Height:</b> {row['Height_m']}m | <b>Diam:</b> {row['Diameter_cm']}cm<br>
                    <b>Location:</b> {row['Barangay']}, {row['Municipality']}<br>
                    <b>Status:</b> {status}<br><br>
                    <a href="https://www.google.com/maps?q={row['Latitude']},{row['Longitude']}" target="_blank"
                       style="background:#2e7d32;color:white;padding:6px 14px;border-radius:20px;
                              text-decoration:none;font-weight:bold;display:inline-block;">📍 Open in Google Maps</a>
                </div>"""
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)
        else:
            map_df['priority_val'] = 0.0
            for _, r in agg.iterrows():
                mask = (map_df['Barangay'] == r['Barangay']) & (map_df['Municipality'] == r['Municipality'])
                map_df.loc[mask, 'priority_val'] = r['Priority_Score']
            for _, row in map_df.iterrows():
                p = row['priority_val']
                color = '#2e7d32' if p < 0.33 else '#f57f17' if p < 0.66 else '#c62828'
                label = '🟢 Low' if p < 0.33 else '🟡 Medium' if p < 0.66 else '🔴 High'
                radius = max(3, row['Diameter_cm'] * 0.4)
                popup_html = f"""<div style="font-family:sans-serif;font-size:13px;min-width:200px;">
                    <b style="font-size:15px;">{row['Tree_ID']}</b><hr style="margin:4px 0;">
                    <b>Species:</b> {row['Species']}<br>
                    <b>Priority:</b> {label} ({p:.2f})<br>
                    <b>Location:</b> {row['Barangay']}, {row['Municipality']}<br><br>
                    <a href="https://www.google.com/maps?q={row['Latitude']},{row['Longitude']}" target="_blank"
                       style="background:#1565c0;color:white;padding:6px 14px;border-radius:20px;
                              text-decoration:none;font-weight:bold;display:inline-block;">📍 Open in Google Maps</a>
                </div>"""
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)

        st_folium(m, use_container_width=True, height=500)
        st.markdown("""
        <div style="display:flex; gap:1.5rem; font-size:0.8rem; justify-content:center;">
            <span><span style="color:#2e7d32; font-weight:bold;">●</span> Alive</span>
            <span><span style="color:#c62828; font-weight:bold;">●</span> Dead</span>
            <span>Click marker → popup → Open in Google Maps</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Priority Areas")
        with st.spinner("Rendering priority chart..."):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            top10 = agg.sort_values('Priority_Score', ascending=False).head(10)
            bars = sns.barplot(data=top10, x='Priority_Score', y='Barangay', hue='Municipality', ax=ax, dodge=False)
            ax.set_xlabel('Priority Score'); ax.set_ylabel('')
            ax.set_title('Top 10 Barangays by Priority Score', fontweight='bold')
            ax.legend(loc='lower right')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Priority Breakdown")
        priority_counts = agg['Priority'].value_counts()
        with st.spinner("Rendering chart..."):
            fig, ax = plt.subplots(figsize=(7, 5))
            wedges, texts, autotexts = ax.pie(
                priority_counts.values, labels=priority_counts.index,
                autopct='%1.1f%%', colors=['#2ecc71', '#f1c40f', '#e74c3c'],
                startangle=90, explode=[0.02]*3,
                textprops={'fontweight': 'bold'}
            )
            ax.set_title('Priority Level Distribution', fontweight='bold')
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Priority Scoring Methodology"):
        st.markdown("""
        **Priority Score = 0.4 × Mortality Rate + 0.3 × (1 - Normalized Height) + 0.3 × (1 - Normalized Age)**  
        - **High Mortality** → higher priority  
        - **Low avg height** → higher priority (young/stunted regeneration)  
        - **Low avg age** → higher priority (early intervention window)
        """)

    top_areas = agg.sort_values('Priority_Score', ascending=False).head(5)
    top_list = "; ".join([f"{r['Barangay']}, {r['Municipality']} (Score: {r['Priority_Score']:.2f})" for _, r in top_areas.iterrows()])
    avg_priority = agg['Priority_Score'].mean()
    show_ai_analysis(f"Based on Caraga Region priority mapping analysis, the top areas needing reforestation are: {top_list}. Explain why these areas score high and recommend targeted interventions for DENR Caraga.", "gis",
        gauge_label="Average Priority Score", gauge_value=avg_priority, gauge_color="#c62828" if avg_priority > 0.5 else "#f57f17" if avg_priority > 0.33 else "#2e7d32")

# ── 9. NEWS REPORT SUMMARY ──
elif page == "📰 News Report Summary":
    total = len(df)
    alive = (df['Survival_Status'] == 'Alive').sum()
    dead = (df['Survival_Status'] == 'Dead').sum()
    survival_rate = alive / total

    mun_stats = df.groupby('Municipality').agg(
        Total_Trees=('Tree_ID', 'count'),
        Alive=('Survival_Status', lambda x: (x == 'Alive').sum()),
        Dead=('Survival_Status', lambda x: (x == 'Dead').sum()),
        Survival_Rate=('Survival_Status', lambda x: (x == 'Alive').mean()),
        Avg_Age=('Age_Years', 'mean'),
        Avg_Height=('Height_m', 'mean')
    ).reset_index().sort_values('Survival_Rate')

    best_mun = mun_stats.loc[mun_stats['Survival_Rate'].idxmax()]
    worst_mun = mun_stats.loc[mun_stats['Survival_Rate'].idxmin()]

    sp_stats = df.groupby('Species').agg(
        Count=('Tree_ID', 'count'),
        Survival_Rate=('Survival_Status', lambda x: (x == 'Alive').mean()),
        Avg_Height=('Height_m', 'mean'),
        Avg_Diameter=('Diameter_cm', 'mean')
    )
    best_sp = sp_stats['Survival_Rate'].idxmax()
    worst_sp = sp_stats['Survival_Rate'].idxmin()

    soil_stats = df.groupby('Soil_Type').agg(
        Survival_Rate=('Survival_Status', lambda x: (x == 'Alive').mean()),
        Count=('Tree_ID', 'count')
    )
    best_soil = soil_stats['Survival_Rate'].idxmax()

    young_trees = df[df['Age_Years'] <= 5]
    mature_trees = df[df['Age_Years'] > 20]
    young_survival = (young_trees['Survival_Status'] == 'Alive').mean()
    mature_survival = (mature_trees['Survival_Status'] == 'Alive').mean() if len(mature_trees) > 0 else 0
    total_carbon = (0.05 * (df['Diameter_cm'] ** 2 * df['Height_m']).sum() * 0.47) / 1000

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Forest Overview")
    cols = st.columns(4)
    cols[0].metric("Total Trees", total)
    cols[1].metric("Alive", f"{alive} ({survival_rate:.1%})")
    cols[2].metric("Dead", f"{dead} ({1-survival_rate:.1%})")
    cols[3].metric("Municipalities", df['Municipality'].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

    status_color, status_icon, status_label = \
        ("#2e7d32", "✅", "Healthy") if survival_rate >= 0.70 else \
        ("#f57f17", "⚠️", "Moderate Concern") if survival_rate >= 0.50 else \
        ("#c62828", "🔴", "Critical")

    st.markdown(f"""
    <div class="card" style="border-left: 5px solid {status_color}; background: {status_color}08;">
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:2rem;">{status_icon}</span>
            <div>
                <strong style="color:{status_color}; font-size:1.1rem;">Forest Status: {status_label}</strong>
                <p style="margin:2px 0 0; color:#666;">
                    {survival_rate:.1%} overall survival rate — 
                    {'Healthy forest ecosystem.' if survival_rate >= 0.70 else 
                     'Some areas need attention.' if survival_rate >= 0.50 else 
                     'Immediate intervention required.'}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Key Insights")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**🏆 Best Municipality:** {best_mun['Municipality']}")
        st.caption(f"{best_mun['Survival_Rate']:.1%} survival — {int(best_mun['Alive'])} alive / {int(best_mun['Total_Trees'])} trees")
        st.markdown(f"**🌿 Best Species:** {best_sp}")
        st.caption(f"{sp_stats.loc[best_sp, 'Survival_Rate']:.1%} survival rate")
        st.markdown(f"**🧪 Best Soil Type:** {best_soil}")
        st.caption(f"{soil_stats.loc[best_soil, 'Survival_Rate']:.1%} survival rate")

    with c2:
        st.markdown(f"**⚠️ Needs Intervention:** {worst_mun['Municipality']}")
        st.caption(f"{worst_mun['Survival_Rate']:.1%} survival — {int(worst_mun['Dead'])} dead / {int(worst_mun['Total_Trees'])} trees")
        st.markdown(f"**🌿 Weakest Species:** {worst_sp}")
        st.caption(f"{sp_stats.loc[worst_sp, 'Survival_Rate']:.1%} survival — may need site-matching")
        st.markdown(f"**🌍 Carbon Stored:** {total_carbon:.1f} t ({(total_carbon * 3.67):.1f} t CO₂e)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Age Structure & Alerts")
    c1, c2 = st.columns(2)
    c1.metric("Young Trees (≤5 yrs)", f"{len(young_trees)} trees", f"{young_survival:.1%} survival")
    c2.metric("Mature Trees (>20 yrs)", f"{len(mature_trees)} trees", f"{mature_survival:.1%} survival")

    has_alert = False
    for _, m in mun_stats.iterrows():
        if m['Survival_Rate'] < 0.50:
            st.error(f"**{m['Municipality']}** — Critical ({m['Survival_Rate']:.1%}). Immediate reforestation needed.")
            has_alert = True
        elif m['Survival_Rate'] < 0.65:
            st.warning(f"**{m['Municipality']}** — Moderate ({m['Survival_Rate']:.1%}). Monitoring recommended.")
            has_alert = True
    if not has_alert:
        st.success("No critical alerts. All municipalities have adequate survival rates.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("View Full Per-Municipality Report"):
        st.dataframe(mun_stats.style.format({
            'Survival_Rate': '{:.1%}', 'Avg_Age': '{:.1f} yrs', 'Avg_Height': '{:.1f} m'
        }), use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Forestry News")
    st.caption("Sourced from Google News — forestry, environment, and DENR updates")

    @st.cache_data(ttl=600)
    def fetch_live_news():
        import feedparser, time, re, urllib.request
        sources = [
            "https://news.google.com/rss/search?q=forestry+Caraga+DENR&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=reforestation+Caraga+environment&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=climate+change+forest+conservation+Caraga&hl=en-PH&gl=PH&ceid=PH:en",
        ]
        seen_titles = set()
        entries = []
        for url in sources:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:5]:
                    t = e.title.strip()
                    if t not in seen_titles:
                        seen_titles.add(t)
                        image_url = None
                        if hasattr(e, 'media_content'):
                            for mc in e.media_content:
                                if mc.get('type', '').startswith('image'):
                                    image_url = mc['url']
                                    break
                        if not image_url and hasattr(e, 'summary'):
                            m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', e.summary)
                            if m:
                                image_url = m.group(1)
                        if not image_url:
                            try:
                                with urllib.request.urlopen(e.link, timeout=2) as resp:
                                    html = resp.read().decode('utf-8', errors='ignore')
                                    m = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
                                    if m:
                                        image_url = m.group(1)
                            except:
                                pass
                        raw_summary = getattr(e, 'summary', '')
                        clean_summary = re.sub(r'<[^>]+>', '', raw_summary)[:200]
                        entries.append({
                            "title": t, "link": e.link,
                            "source": getattr(e, "source", {}).get("title", "Google News") if hasattr(e, "source") else "Google News",
                            "published": getattr(e, "published", "Just now"),
                            "summary": clean_summary, "image_url": image_url,
                        })
            except:
                pass
            time.sleep(0.3)
        return entries[:10]

    if st.button("Refresh Live News", key="refresh_news"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching live forestry news..."):
        news_entries = fetch_live_news()
    if news_entries:
        for ne in news_entries:
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    if ne.get('image_url'):
                        st.image(ne['image_url'], use_container_width=True)
                with cols[1]:
                    st.markdown(f"**[{ne['title']}]({ne['link']})**")
                    st.caption(f"{ne['source']} • {ne['published']}")
                    if ne['summary']:
                        st.markdown(ne['summary'])
    else:
        st.info("Live news feed unavailable at this time.")
    st.markdown('</div>', unsafe_allow_html=True)

    show_ai_analysis(f"Summarize this Caraga Region forest health report: {total} trees across {df['Municipality'].nunique()} municipalities. Survival rate: {survival_rate:.1%}. Best municipality: {best_mun['Municipality']} ({best_mun['Survival_Rate']:.1%}), worst: {worst_mun['Municipality']} ({worst_mun['Survival_Rate']:.1%}). Best species: {best_sp} ({sp_stats.loc[best_sp, 'Survival_Rate']:.1%}), weakest: {worst_sp} ({sp_stats.loc[worst_sp, 'Survival_Rate']:.1%}). Best soil: {best_soil}. Give executive recommendations for DENR Caraga action.", "news",
        gauge_label="Overall Forest Health Score", gauge_value=survival_rate, gauge_color="#2e7d32" if survival_rate >= 0.7 else "#f57f17" if survival_rate >= 0.5 else "#c62828")

    st.caption("Report generated from forest inventory data • For DENR decision support")

# ── 10. REFORESTATION PROJECT MONITORING DATASET ──
elif page == "📋 Reforestation Project Monitoring Dataset":
    _base = os.path.dirname(os.path.abspath(__file__))
    REFORESTATION_DATA_PATH = os.path.join(_base, 'data', 'reforestation_projects_1000.csv')
    MODEL_PATH = os.path.join(_base, 'models', 'project_status_model.pkl')
    PREPROC_PATH = os.path.join(_base, 'models', 'preprocessor.pkl')
    LABEL_PATH = os.path.join(_base, 'models', 'label_encoder.pkl')
    FEATURES_PATH = os.path.join(_base, 'models', 'feature_columns.pkl')

    @st.cache_data
    def load_reforestation_data():
        return pd.read_csv(REFORESTATION_DATA_PATH)

    def load_reforestation_artifacts():
        try:
            model = joblib.load(MODEL_PATH)
            preproc = joblib.load(PREPROC_PATH)
            le = joblib.load(LABEL_PATH)
            feats = joblib.load(FEATURES_PATH)
            return model, preproc, le, feats
        except:
            return None, None, None, None

    rdf = load_reforestation_data()
    rf_model_r, rf_preproc, rf_label, rf_features = load_reforestation_artifacts()
    models_ready = rf_model_r is not None

    if not models_ready:
        st.error("Models not found. Please run Reforestation_Model.ipynb first to train and save the models.")
        st.markdown("""```bash\njupyter nbconvert --to notebook --execute Reforestation_Model.ipynb\n```""")
        st.stop()

    total_projects = len(rdf)
    success_count = (rdf['Project_Status'] == 'Successful').sum()
    moderate_count = (rdf['Project_Status'] == 'Moderate').sum()
    failed_count = (rdf['Project_Status'] == 'Failed').sum()
    total_seedlings_planted = int(rdf['Planted_Seedlings'].sum())
    total_funding = int(rdf['Funding_PHP'].sum())
    total_pest = int(rdf['Pest_Incidents'].sum())
    total_fire = int(rdf['Fire_Incidents'].sum())
    avg_survival = rdf['Survival_Rate'].mean()

    color_map = {"Successful": "#2e7d32", "Moderate": "#f39c12", "Failed": "#c62828"}
    icon_map = {"Successful": "✅", "Moderate": "⚠️", "Failed": "❌"}
    palette_status = {'Successful': '#2ecc71', 'Moderate': '#f39c12', 'Failed': '#e74c3c'}

    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
        <div style="font-size:1.8rem;">📋</div>
        <div>
            <h2 style="margin:0; font-size:1.3rem; color:#1b5e20;">Reforestation Project Monitoring Dashboard</h2>
            <p style="margin:0; color:#6c757d; font-size:0.8rem;">AI-Powered Analysis of Reforestation Project Outcomes</p>
        </div>
    </div>
    <hr style="margin:0.5rem 0; opacity:0.2;">
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 1. KPI CARDS (6)
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cols = st.columns(6)
    cols[0].markdown(f"""<div class="metric-card" style="border-left-color:#2e7d32"><div class="label">🌱 Total Projects</div><div class="value">{total_projects}</div></div>""", unsafe_allow_html=True)
    cols[1].markdown(f"""<div class="metric-card" style="border-left-color:#1565c0"><div class="label">🌳 Seedlings Planted</div><div class="value">{total_seedlings_planted:,}</div></div>""", unsafe_allow_html=True)
    cols[2].markdown(f"""<div class="metric-card" style="border-left-color:#00897b"><div class="label">📈 Avg Survival Rate</div><div class="value">{avg_survival:.1%}</div></div>""", unsafe_allow_html=True)
    cols[3].markdown(f"""<div class="metric-card" style="border-left-color:#6a1b9a"><div class="label">💰 Total Funding</div><div class="value">₱{total_funding:,}</div></div>""", unsafe_allow_html=True)
    cols[4].markdown(f"""<div class="metric-card" style="border-left-color:#d84315"><div class="label">🔥 Fire Incidents</div><div class="value">{total_fire}</div></div>""", unsafe_allow_html=True)
    cols[5].markdown(f"""<div class="metric-card" style="border-left-color:#f9a825"><div class="label">🐛 Pest Incidents</div><div class="value">{total_pest}</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 2. PROJECT STATUS DISTRIBUTION  +  SURVIVAL BY MUNICIPALITY
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Project Status Distribution")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        status_counts = rdf['Project_Status'].value_counts()
        status_colors = [palette_status[s] for s in status_counts.index]
        wedges, texts, autotexts = ax.pie(
            status_counts.values, labels=status_counts.index,
            autopct='%1.1f%%', colors=status_colors,
            startangle=90, explode=[0.02]*len(status_counts),
            textprops={'fontweight': 'bold', 'fontsize': 11}
        )
        ax.set_title('Project Status Distribution', fontweight='bold', fontsize=13)
        st.pyplot(fig)
    with c2:
        st.subheader("🌱 Survival Rate by Municipality")
        mun_survival = rdf.groupby('Municipality')['Survival_Rate'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars = ax.barh(mun_survival.index, mun_survival.values,
                       color=[palette_status.get('Successful') if v >= 0.6
                              else palette_status.get('Moderate') if v >= 0.4
                              else palette_status.get('Failed') for v in mun_survival.values],
                       edgecolor='white', height=0.7)
        for bar, val in zip(bars, mun_survival.values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.0%}', va='center', fontweight='bold', fontsize=10)
        ax.set_xlabel('Survival Rate', fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Average Survival Rate by Municipality', fontweight='bold', fontsize=13)
        ax.set_xlim(0, 1.05)
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 3. FUNDING vs SURVIVAL  +  RAINFALL vs SURVIVAL
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 Funding vs Survival Rate")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for status, color in [('Successful', '#2ecc71'), ('Moderate', '#f39c12'), ('Failed', '#e74c3c')]:
            subset = rdf[rdf['Project_Status'] == status]
            ax.scatter(subset['Funding_PHP'], subset['Survival_Rate'],
                       alpha=0.4, s=25, color=color, label=status, edgecolors='white', linewidth=0.4)
        ax.set_xlabel('Funding (PHP)', fontweight='bold')
        ax.set_ylabel('Survival Rate', fontweight='bold')
        ax.set_title('Funding vs Survival Rate', fontweight='bold', fontsize=13)
        ax.legend()
        st.pyplot(fig)
    with c2:
        st.subheader("🌧 Rainfall vs Survival Rate")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for status, color in [('Successful', '#2ecc71'), ('Moderate', '#f39c12'), ('Failed', '#e74c3c')]:
            subset = rdf[rdf['Project_Status'] == status]
            ax.scatter(subset['Rainfall_mm'], subset['Survival_Rate'],
                       alpha=0.4, s=25, color=color, label=status, edgecolors='white', linewidth=0.4)
        ax.set_xlabel('Rainfall (mm)', fontweight='bold')
        ax.set_ylabel('Survival Rate', fontweight='bold')
        ax.set_title('Rainfall vs Survival Rate', fontweight='bold', fontsize=13)
        ax.legend()
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 4. TEMPERATURE vs SURVIVAL  +  MONITORING VISITS vs SURVIVAL
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌡 Temperature vs Survival Rate")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for status, color in [('Successful', '#2ecc71'), ('Moderate', '#f39c12'), ('Failed', '#e74c3c')]:
            subset = rdf[rdf['Project_Status'] == status]
            ax.scatter(subset['Temperature_C'], subset['Survival_Rate'],
                       alpha=0.4, s=25, color=color, label=status, edgecolors='white', linewidth=0.4)
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Survival Rate', fontweight='bold')
        ax.set_title('Temperature vs Survival Rate', fontweight='bold', fontsize=13)
        ax.legend()
        st.pyplot(fig)
    with c2:
        st.subheader("👀 Monitoring Visits vs Survival")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for status, color in [('Successful', '#2ecc71'), ('Moderate', '#f39c12'), ('Failed', '#e74c3c')]:
            subset = rdf[rdf['Project_Status'] == status]
            ax.scatter(subset['Monitoring_Visits'], subset['Survival_Rate'],
                       alpha=0.4, s=25, color=color, label=status, edgecolors='white', linewidth=0.4)
        ax.set_xlabel('Monitoring Visits', fontweight='bold')
        ax.set_ylabel('Survival Rate', fontweight='bold')
        ax.set_title('Monitoring Visits vs Survival Rate', fontweight='bold', fontsize=13)
        ax.legend()
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 5. PEST INCIDENTS  +  FIRE INCIDENTS  (by Municipality)
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🐛 Pest Incidents by Municipality")
        mun_pest = rdf.groupby('Municipality')['Pest_Incidents'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_pest = [palette_status.get('Failed') if v >= 3
                       else palette_status.get('Moderate') for v in mun_pest.values]
        ax.barh(range(len(mun_pest)), mun_pest.values, color=colors_pest, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(mun_pest)))
        ax.set_yticklabels(mun_pest.index)
        ax.set_xlabel('Average Pest Incidents', fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Pest Incident Hotspots by Municipality', fontweight='bold', fontsize=13)
        for i, v in enumerate(mun_pest.values):
            ax.text(v + 0.1, i, f'{v:.1f}', va='center', fontweight='bold')
        st.pyplot(fig)
    with c2:
        st.subheader("🔥 Fire Incidents by Municipality")
        mun_fire = rdf.groupby('Municipality')['Fire_Incidents'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_fire = [palette_status.get('Failed') if v >= 1.0
                       else palette_status.get('Moderate') if v >= 0.5
                       else palette_status.get('Successful') for v in mun_fire.values]
        ax.barh(range(len(mun_fire)), mun_fire.values, color=colors_fire, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(mun_fire)))
        ax.set_yticklabels(mun_fire.index)
        ax.set_xlabel('Average Fire Incidents', fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Fire Incident Hotspots by Municipality', fontweight='bold', fontsize=13)
        for i, v in enumerate(mun_fire.values):
            ax.text(v + 0.05, i, f'{v:.2f}', va='center', fontweight='bold')
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 6. SOIL TYPE DISTRIBUTION  +  SURVIVAL BY SOIL TYPE
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌍 Soil Type Distribution")
        soil_counts = rdf['Soil_Type'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        soil_colors_pie = ['#4CAF50', '#795548', '#FFC107', '#607D8B', '#9E9E9E'][:len(soil_counts)]
        wedges, texts, autotexts = ax.pie(
            soil_counts.values, labels=soil_counts.index,
            autopct='%1.1f%%', colors=soil_colors_pie,
            startangle=90, explode=[0.02]*len(soil_counts),
            textprops={'fontweight': 'bold'}
        )
        ax.set_title('Soil Type Distribution', fontweight='bold', fontsize=13)
        st.pyplot(fig)
    with c2:
        st.subheader("🌿 Survival Rate by Soil Type")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        order = rdf.groupby('Soil_Type')['Survival_Rate'].mean().sort_values(ascending=False).index
        sns.boxplot(data=rdf, x='Soil_Type', y='Survival_Rate', order=order, ax=ax,
                    palette='Set2', hue='Soil_Type', legend=False, width=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('Survival Rate', fontweight='bold')
        ax.set_title('Survival Rate Distribution by Soil Type', fontweight='bold', fontsize=13)
        means = rdf.groupby('Soil_Type')['Survival_Rate'].mean()
        for i, soil in enumerate(order):
            ax.text(i, means[soil] + 0.02, f'{means[soil]:.0%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 7. YEARLY PERFORMANCE  +  TARGET vs PLANTED (grouped bar)
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Yearly Performance Trend")
        yearly = rdf.groupby('Year')['Survival_Rate'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(yearly['Year'], yearly['Survival_Rate'], marker='o', linewidth=2.5,
                color='#2e7d32', markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(yearly['Year'], yearly['Survival_Rate'], alpha=0.15, color='#2e7d32')
        for _, row in yearly.iterrows():
            ax.text(row['Year'], row['Survival_Rate'] + 0.02, f'{row["Survival_Rate"]:.0%}',
                    ha='center', fontweight='bold', fontsize=9)
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Average Survival Rate', fontweight='bold')
        ax.set_title('Yearly Reforestation Performance Trend', fontweight='bold', fontsize=13)
        ax.set_xticks(yearly['Year'])
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    with c2:
        st.subheader("🌳 Target vs Planted Seedlings")
        mun_seedlings = rdf.groupby('Municipality')[['Target_Seedlings', 'Planted_Seedlings']].mean()
        mun_seedlings = mun_seedlings.sort_values('Target_Seedlings', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(mun_seedlings))
        w = 0.35
        bars1 = ax.bar(x - w/2, mun_seedlings['Target_Seedlings'], w, label='Target',
                       color='#43a047', edgecolor='white')
        bars2 = ax.bar(x + w/2, mun_seedlings['Planted_Seedlings'], w, label='Planted',
                       color='#66bb6a', edgecolor='white')
        ax.set_xlabel('')
        ax.set_ylabel('Number of Seedlings', fontweight='bold')
        ax.set_title('Target vs Planted Seedlings (Top 10)', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(mun_seedlings.index, rotation=45, ha='right')
        ax.legend()
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 8. FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Feature Importance (Random Forest)")
    st.markdown("Key variables driving project status predictions, ranked by impact.")
    with st.spinner("Computing feature importance..."):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            _rdf_imp = rdf.drop(columns=['Project_ID']).copy()
            _le_mun = LabelEncoder()
            _le_soil = LabelEncoder()
            _rdf_imp['Municipality'] = _le_mun.fit_transform(_rdf_imp['Municipality'])
            _rdf_imp['Soil_Type'] = _le_soil.fit_transform(_rdf_imp['Soil_Type'])
            _le_target = LabelEncoder()
            _y_imp = _le_target.fit_transform(_rdf_imp['Project_Status'])
            _X_imp = _rdf_imp.drop(columns=['Project_Status'])
            _rf_imp = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            _rf_imp.fit(_X_imp, _y_imp)
            _importances = _rf_imp.feature_importances_
            _feat_names = _X_imp.columns.tolist()
            _sorted_idx = np.argsort(_importances)[::-1]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            _top_n = min(12, len(_feat_names))
            _top_feats = [_feat_names[i] for i in _sorted_idx[:_top_n]][::-1]
            _top_vals = [_importances[i] for i in _sorted_idx[:_top_n]][::-1]
            _imp_colors = plt.cm.Greens(np.linspace(0.35, 0.85, _top_n))
            bars = ax.barh(range(_top_n), _top_vals, color=_imp_colors, edgecolor='white', height=0.65)
            for i, (bar, val) in enumerate(zip(bars, _top_vals)):
                ax.text(bar.get_width() + 0.005, i, f'{val:.1%}',
                        va='center', fontweight='bold', fontsize=10)
            ax.set_yticks(range(_top_n))
            ax.set_yticklabels(_top_feats, fontsize=10)
            ax.set_xlabel('Importance Score', fontweight='bold')
            ax.set_title('Feature Importance — Top Factors Affecting Project Status', fontweight='bold', fontsize=13)
            ax.set_xlim(0, max(_top_vals) * 1.25)
            st.pyplot(fig)
        except Exception as _e:
            st.info(f"Feature importance unavailable: {_e}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 9. AI PREDICTION FORM
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔮 Predict Project Status")
    st.markdown("Enter project details to predict the outcome using machine learning.")
    col1, col2 = st.columns(2)
    with col1:
        municipality = st.selectbox("Municipality", rdf['Municipality'].unique(), key='rf_mun')
        survival_rate = st.slider("Survival Rate", 0.0, 1.0, 0.65, 0.01, key='rf_sr')
        funding = st.number_input("Funding (PHP)", 30000, 3000000, 500000, key='rf_fund')
    with col2:
        rainfall = st.slider("Rainfall (mm)", 300, 5000, 2500, key='rf_rain')
        temperature = st.slider("Temperature (°C)", 20.0, 38.0, 28.0, 0.5, key='rf_temp')
        monitoring = st.slider("Monitoring Visits", 0, 30, 5, key='rf_mon')
        soil_type = st.selectbox("Soil Type", rdf['Soil_Type'].unique(), key='rf_soil')
        pest = st.slider("Pest Incidents", 0, 15, 2, key='rf_pest')
        fire = st.slider("Fire Incidents", 0, 10, 0, key='rf_fire')

    ml_model_choice = st.selectbox(
        "Choose ML Model",
        ["Decision Tree", "Random Forest", "Gradient Boosting", "KNN", "Logistic Regression"],
        key='rf_ml_choice'
    )

    if st.button("Predict Project Status", type="primary", key='rf_predict'):
        target_seedlings = int(rdf['Target_Seedlings'].median())
        planted_seedlings = int(rdf['Planted_Seedlings'].median())
        inp_df = pd.DataFrame([[
            2025, municipality, target_seedlings, planted_seedlings, survival_rate,
            funding, rainfall, temperature, monitoring, soil_type, pest, fire
        ]], columns=rf_features)

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        model_classes = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42)
        }
        temp_pipeline = Pipeline([
            ('preprocessor', rf_preproc),
            ('classifier', model_classes[ml_model_choice])
        ])
        rdf_X = rdf.drop(columns=['Project_ID', 'Project_Status'])
        rdf_y = rf_label.transform(rdf['Project_Status'])
        temp_pipeline.fit(rdf_X, rdf_y)

        pred_encoded = temp_pipeline.predict(inp_df)[0]
        pred_label = rf_label.inverse_transform([pred_encoded])[0]
        proba = temp_pipeline.predict_proba(inp_df)[0]
        confidence = float(max(proba))

        pc = color_map.get(pred_label, "#666")
        pi = icon_map.get(pred_label, "❓")

        st.markdown("---")
        st.subheader("🧠 AI Prediction Result")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card" style="border-left-color:{pc}"><div class="label">Predicted Status</div><div class="value">{pi} {pred_label}</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card" style="border-left-color:#1976d2"><div class="label">Confidence</div><div class="value">{confidence:.1%}</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card" style="border-left-color:{pc}"><div class="label">Model Used</div><div class="value">{ml_model_choice}</div></div>""", unsafe_allow_html=True)
        st.progress(float(confidence))

        with st.expander("Class Probabilities"):
            for i, cls in enumerate(rf_label.classes_):
                pct = proba[i]
                cls_color = color_map.get(cls, "#666")
                st.markdown(f"""<div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
                    <span style="width:100px; font-weight:600; color:{cls_color};">{cls}</span>
                    <div style="flex:1; height:20px; background:#e9ecef; border-radius:10px; overflow:hidden;">
                        <div style="height:100%; width:{pct:.1%}; background:{cls_color}; border-radius:10px;"></div>
                    </div>
                    <span style="width:50px; text-align:right; font-weight:600;">{pct:.1%}</span>
                </div>""", unsafe_allow_html=True)

        _pred_gauge = confidence if pred_label == "Successful" else (1 - confidence) if pred_label == "Failed" else confidence * 0.5 + 0.25
        show_ai_analysis(
            f"A reforestation project in {municipality}, Caraga Region, with survival_rate={survival_rate:.2f}, funding={funding} PHP, rainfall={rainfall}mm, temperature={temperature:.1f}°C, monitoring_visits={monitoring}, soil_type={soil_type}, pest_incidents={pest}, fire_incidents={fire} was predicted as **{pred_label}** with {confidence:.1%} confidence using {ml_model_choice}."
            "\n\nExplain the key factors that influenced this prediction. Discuss what went well or what went wrong, and provide actionable recommendations for improving future project outcomes in this municipality.",
            "reforestation_pred",
            gauge_label="Project Success Score", gauge_value=_pred_gauge, gauge_color=pc
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 10. AI PROJECT SUMMARIZATION
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Project Summarization")
    ai_model_rf = st.selectbox("Select AI Model for Summary", AI_MODELS, key='rf_ai_model')

    _mun_perf = rdf.groupby('Municipality').agg(
        Total_Projects=('Project_ID', 'count'),
        Success_Rate=('Project_Status', lambda x: (x == 'Successful').mean()),
        Avg_Survival=('Survival_Rate', 'mean')
    ).reset_index().sort_values('Success_Rate', ascending=False)
    _best_mun = _mun_perf.iloc[0]
    _worst_mun = _mun_perf.iloc[-1]

    summary_prompt = (
        f"{AI_REGION_CONTEXT}\n\n"
        f"Generate a comprehensive natural language summary of the Caraga Region reforestation project portfolio based on this data:\n\n"
        f"- Total projects analyzed: {total_projects}\n"
        f"- Municipalities covered: {rdf['Municipality'].nunique()}\n"
        f"- Project status breakdown: {success_count} Successful ({success_count/total_projects:.1%}), "
        f"{moderate_count} Moderate ({moderate_count/total_projects:.1%}), "
        f"{failed_count} Failed ({failed_count/total_projects:.1%})\n"
        f"- Average survival rate across all projects: {avg_survival:.2%}\n"
        f"- Total seedlings planted: {total_seedlings_planted:,}\n"
        f"- Average funding per project: ₱{rdf['Funding_PHP'].mean():,.0f}\n"
        f"- Average monitoring visits: {rdf['Monitoring_Visits'].mean():.1f}\n"
        f"- Average pest incidents: {rdf['Pest_Incidents'].mean():.1f}\n"
        f"- Average fire incidents: {rdf['Fire_Incidents'].mean():.2f}\n"
        f"- Best performing municipality: {_best_mun['Municipality']} ({_best_mun['Success_Rate']:.1%} success)\n"
        f"- Lowest performing municipality: {_worst_mun['Municipality']} ({_worst_mun['Success_Rate']:.1%} success)\n\n"
        f"Write in a professional report style. Include:\n"
        f"1. Executive summary of overall project health\n"
        f"2. Key factors driving success (high survival rate, adequate funding, monitoring, etc.)\n"
        f"3. Key risk factors leading to failure (pest outbreaks, fire incidents, low funding, etc.)\n"
        f"4. Municipal-level comparisons and insights\n"
        f"5. Specific actionable recommendations for DENR Caraga to improve reforestation project outcomes\n\n"
        f"Conclude with a 'SUMMARY ASSESSMENT' section stating whether the overall reforestation program is on track."
    )

    if st.button("Generate AI Project Summary", type="primary", key='rf_summary'):
        with st.spinner("Generating comprehensive project summary with AI..."):
            summary = cached_ollama(summary_prompt, ai_model_rf)
        if summary:
            st.markdown("---")
            st.markdown(summary)
        else:
            st.info("AI summary unavailable. Ensure Ollama is running.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 11. GIS PRIORITY MAP
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺 GIS Priority Map")
    st.markdown("Municipality-level project status overview.")
    _municipality_coords = {
        "Bunawan": (8.239, 125.990), "Talacogon": (8.456, 125.784),
        "San Francisco": (8.505, 125.975), "Lianga": (8.633, 126.094),
        "Rosario": (8.380, 126.000), "Bislig": (8.215, 126.316),
        "Trento": (8.046, 126.063), "Veruela": (8.073, 125.956),
        "Loreto": (8.187, 125.853), "Lanuza": (9.234, 126.058),
        "Cortes": (9.276, 126.191), "Cagwait": (8.918, 126.302),
        "Barobo": (8.578, 126.204), "Hinatuan": (8.371, 126.338),
        "Lingig": (8.038, 126.412)
    }
    _mun_gis = rdf.groupby('Municipality').agg(
        Total=('Project_ID', 'count'),
        Status=('Project_Status', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Moderate'),
        Survival=('Survival_Rate', 'mean')
    ).reset_index()
    _mun_gis['Lat'] = _mun_gis['Municipality'].map(lambda m: _municipality_coords.get(m, (8.5, 126.0))[0])
    _mun_gis['Lon'] = _mun_gis['Municipality'].map(lambda m: _municipality_coords.get(m, (8.5, 126.0))[1])
    _center_lat = _mun_gis['Lat'].mean()
    _center_lon = _mun_gis['Lon'].mean()

    _m = folium.Map(location=[_center_lat, _center_lon], zoom_start=9,
                    tiles='CartoDB Positron', control_scale=True)
    for _, _r in _mun_gis.iterrows():
        _c = color_map.get(_r['Status'], '#999')
        folium.CircleMarker(
            location=[_r['Lat'], _r['Lon']],
            radius=8 + _r['Total'] * 0.3, color=_c, fill=True,
            fill_color=_c, fill_opacity=0.7, weight=2,
            popup=folium.Popup(
                f"<b>{_r['Municipality']}</b><br>"
                f"Status: {_r['Status']}<br>"
                f"Projects: {int(_r['Total'])}<br>"
                f"Avg Survival: {_r['Survival']:.1%}<br>"
                f"<span style='color:{_c}; font-weight:bold;'>● {_r['Status']}</span>",
                max_width=250)
        ).add_to(_m)
    st_folium(_m, use_container_width=True, height=420)
    st.markdown("""
    <div style="display:flex; gap:1.5rem; font-size:0.8rem; justify-content:center; margin-bottom:0.5rem;">
        <span><span style="color:#2e7d32;font-weight:bold;">●</span> Successful</span>
        <span><span style="color:#f39c12;font-weight:bold;">●</span> Moderate</span>
        <span><span style="color:#c62828;font-weight:bold;">●</span> Failed</span>
        <span>Marker size = number of projects</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 12. PROJECT DETAILS TABLE
    # ═══════════════════════════════════════════════════════════
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Project Details Table")
    st.markdown("Search and filter individual project records.")
    _filter_status = st.multiselect("Filter by Status", ["Successful", "Moderate", "Failed"],
                                     default=["Successful", "Moderate", "Failed"])
    _filter_soil = st.multiselect("Filter by Soil Type", rdf['Soil_Type'].unique().tolist(),
                                   default=rdf['Soil_Type'].unique().tolist())
    _search = st.text_input("Search by Municipality or Project ID", placeholder="Type to search...")
    _df_display = rdf[rdf['Project_Status'].isin(_filter_status) & rdf['Soil_Type'].isin(_filter_soil)]
    if _search:
        _df_display = _df_display[
            _df_display['Municipality'].str.contains(_search, case=False, na=False) |
            _df_display['Project_ID'].str.contains(_search, case=False, na=False)
        ]
    st.dataframe(_df_display.style.map(
        lambda s: f'background-color: {color_map.get(s, "")}20; color: {color_map.get(s, "#000")}; font-weight: bold;'
        if s in color_map else '', subset=['Project_Status']
    ), use_container_width=True, height=350)
    st.markdown(f"**Showing {len(_df_display)} of {total_projects} projects**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("Powered by Machine Learning • Reforestation Project Monitoring System v1.0")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding:0.8rem; background:#f8f9fa; border-radius:8px; font-size:0.75rem; color:#6c757d;">
    <strong>About</strong><br>
    Built with Streamlit • Random Forest models trained on forest inventory data.
</div>
""", unsafe_allow_html=True)

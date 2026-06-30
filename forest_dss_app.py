import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, time, re, urllib.request, feedparser
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
import xgboost as xgb

st.set_page_config(page_title="Forest DSS", layout="wide", page_icon="🌳")

# ── CUSTOM CSS ──
st.markdown("""
<style>
    .main > .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 1200px; }
    .stApp { background-color: #f8f9fa; }
    .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1rem; border: 1px solid #e9ecef; }
    .metric-card { background: white; border-radius: 10px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); text-align: center; }
    .sidebar-header { background: linear-gradient(135deg, #1b5e20, #2e7d32); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center; }
    .sidebar-header h3 { margin: 0; font-size: 1rem; color: white !important; }
    .sidebar-header p { margin: 4px 0 0; font-size: 0.75rem; opacity: 0.85; }
    div[data-testid="stMetric"] { background: white; padding: 0.8rem 1rem; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-left: 3px solid #2e7d32; }
    .stButton button { border-radius: 8px; font-weight: 600; transition: all 0.2s; }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .rec-box { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .ai-box { background: #e3f2fd; border-left: 4px solid #1565c0; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .chat-msg { padding: 0.6rem 1rem; border-radius: 16px; margin: 0.3rem 0; max-width: 85%; }
    .chat-user { background: #e3f2fd; margin-left: auto; border-bottom-right-radius: 4px; }
    .chat-ai { background: #f5f5f5; margin-right: auto; border-bottom-left-radius: 4px; }
    hr { margin: 1rem 0; opacity: 0.2; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = 'forest_inventory_dataset_1000 - forest_inventory_dataset_1000.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ── HEADER ──
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <div style="font-size:2rem;">🌳</div>
    <div>
        <h1 style="margin:0; font-size:1.6rem; color:#1b5e20;">DENR Forest Decision Support System</h1>
        <p style="margin:0; color:#6c757d; font-size:0.85rem;">AI-Powered Analytics for Sustainable Forest Management</p>
    </div>
</div>
<hr style="margin:0.8rem 0;">
""", unsafe_allow_html=True)

# ── SIDEBAR ──
st.sidebar.markdown('<div class="sidebar-header"><h3>🌳 Navigation</h3><p>Forest DSS v2.0 — AI Powered</p></div>', unsafe_allow_html=True)

PAGES = [
    "📊 Data Overview", "🌱 Survival Prediction", "⚠️ Mortality Risk",
    "🌿 Species Recommendation", "📈 Growth Prediction", "🌳 Carbon Storage",
    "📦 Timber Volume", "🗺️ GIS Priority Mapping", "🤖 AI Assistant",
    "📰 News Report Summary",
]
page = st.sidebar.radio("", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model")
model_names = list(models.keys())
idx = model_names.index(st.session_state.model_choice) if st.session_state.model_choice in model_names else 0
st.session_state.model_choice = st.sidebar.selectbox(
    "Prediction Engine",
    model_names,
    index=idx,
    key="model_selector",
    help="Switch between ML models for predictions"
)
for name, acc in model_accs.items():
    check = "✅" if name == st.session_state.model_choice else ""
    st.sidebar.caption(f"{check} {name}: {acc:.1%}")

# ── OLLAMA HELPER ──
@st.cache_data(show_spinner=False)
def ask_ollama(prompt, model="qwen3:4b", system=""):
    import ollama
    try:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = ollama.chat(model=model, messages=msgs, options={"num_predict": 1024, "temperature": 0.3})
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ AI unavailable: {e}"

def stream_ollama(prompt, model="qwen3:4b", system=""):
    import ollama
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    stream = ollama.chat(model=model, messages=msgs, stream=True, options={"num_predict": 2048, "temperature": 0.3})
    for chunk in stream:
        yield chunk["message"]["content"]

# ── MODEL TRAINING ──
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

    models = {}
    models["Random Forest"] = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    models["Random Forest"].fit(X_train, y_train)

    models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
    models["Logistic Regression"].fit(X_train, y_train)

    models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    models["XGBoost"].fit(X_train, y_train)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rf = models["Random Forest"]
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

    model_accs = {}
    for name, m in models.items():
        model_accs[name] = accuracy_score(y_test, m.predict(X_test))

    return models, enc, X.columns.tolist(), X_test, y_test, cv_scores, imp, model_accs

models, encoders, feature_cols, X_test, y_test, cv_scores, perm_imp, model_accs = train_survival_model()

# ── Model Switcher ──
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Random Forest"

def get_model():
    return models[st.session_state.model_choice]

sns.set_style("whitegrid")
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})

# ── HELPER FUNCTIONS ──
def make_input_row(species, barangay, municipality, lat, lng, age, height, diameter, soil):
    return pd.DataFrame([[species, barangay, municipality, lat, lng, age, height, diameter, soil]],
        columns=['Species', 'Barangay', 'Municipality', 'Latitude', 'Longitude', 'Age_Years', 'Height_m', 'Diameter_cm', 'Soil_Type'])

def encode_input(inp):
    for c in ['Species', 'Barangay', 'Municipality', 'Soil_Type']:
        inp[c] = encoders[c].transform(inp[c])
    return inp

# ══════════════════════════════════════════════════════════════
#  1. DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Data Overview":
    alive_count = (df['Survival_Status'] == 'Alive').sum()
    dead_count = (df['Survival_Status'] == 'Dead').sum()
    surv_rate = alive_count / len(df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Dataset Overview")
    cols = st.columns(6)
    cols[0].metric("Total Trees", df.shape[0])
    cols[1].metric("Species", df['Species'].nunique())
    cols[2].metric("Municipalities", df['Municipality'].nunique())
    cols[3].metric("Alive", f"{alive_count} ({surv_rate:.1%})")
    cols[4].metric("Dead", f"{dead_count} ({1-surv_rate:.1%})")
    cols[5].metric("Avg Height", f"{df['Height_m'].mean():.1f} m")
    st.markdown('</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Survival by Species")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.countplot(data=df, y='Species', hue='Survival_Status', ax=ax,
                      palette={'Alive': '#2e7d32', 'Dead': '#c62828'})
        ax.set_xlabel('Count'); ax.set_ylabel(''); ax.legend(loc='lower right')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Survival by Soil Type")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.countplot(data=df, x='Soil_Type', hue='Survival_Status', ax=ax,
                      palette={'Alive': '#2e7d32', 'Dead': '#c62828'})
        ax.set_xlabel(''); ax.set_ylabel('Count'); ax.legend(loc='upper right')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🤖 AI Dataset Summary"):
        with st.spinner("Analyzing dataset..."):
            summary = df.describe().to_string()
            prompt = f"""Analyze this Philippine forest inventory dataset summary and provide key insights:
{summary}

Total trees: {df.shape[0]}, Species: {df['Species'].nunique()}, Municipalities: {df['Municipality'].nunique()}
Survival rate: {surv_rate:.1%}

Give a concise 3-4 sentence analysis of forest health, notable patterns, and recommendations."""
            resp = ask_ollama(prompt)
            st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

    with st.expander("View Raw Data & Statistics"):
        tab1, tab2 = st.tabs(["Data Sample", "Statistics"])
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
        with tab2:
            st.dataframe(df.describe(), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  2. SURVIVAL PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🌱 Survival Prediction":
    st.markdown('<div class="card"><h3>🌱 Tree Survival Prediction</h3><p>Enter tree details to predict survival. Toggle <b>What-If Mode</b> to simulate scenarios.</p></div>', unsafe_allow_html=True)

    whatif = st.toggle("🔀 What-If Mode — adjust parameters to see impact", value=False)

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

    inp = encode_input(make_input_row(species, barangay, municipality, lat, lng, age, height, diameter, soil))
    prob = get_model().predict_proba(inp)[0, 1]
    pred = get_model().predict(inp)[0]

    if whatif:
        st.markdown("---")
        st.subheader("🔀 What-If Simulation")
        st.caption("Adjust age or height to see how survival probability changes")
        sim_age = st.slider("Simulate Age", 1, 50, age, key="sim_age")
        sim_ht = st.slider("Simulate Height (m)", 0.5, 40.0, height, key="sim_ht")
        sim_dia = st.slider("Simulate Diameter (cm)", 1.0, 60.0, diameter, key="sim_dia")

        ages = np.arange(1, 51)
        probs = []
        for a in ages:
            inp_sim = encode_input(make_input_row(species, barangay, municipality, lat, lng, a, sim_ht, sim_dia, soil))
            probs.append(get_model().predict_proba(inp_sim)[0, 1])

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(ages, probs, color='#2e7d32', linewidth=2.5)
        ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Survival threshold')
        ax.axvline(age, color='gray', ls='--', alpha=0.5, label='Current age')
        ax.scatter([age], [prob], color='#1565c0', s=120, zorder=5, edgecolors='white')
        ax.set_xlabel('Age (Years)'); ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Probability vs Age', fontweight='bold')
        ax.legend(); ax.set_ylim(0, 1)
        st.pyplot(fig)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {'#2e7d32' if pred else '#c62828'}">
        <div class="label">Prediction</div>
        <div style="font-size:1.6rem;font-weight:700;">{'✅ Alive' if pred else '❌ Dead'}</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #1976d2">
        <div class="label">Survival Probability</div>
        <div style="font-size:1.6rem;font-weight:700;">{prob:.1%}</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {'#2e7d32' if prob > 0.5 else '#c62828'}">
        <div class="label">Confidence</div>
        <div style="font-size:1.6rem;font-weight:700;">{'High' if prob > 0.75 else 'Medium' if prob > 0.5 else 'Low'}</div>
    </div>""", unsafe_allow_html=True)
    st.progress(float(prob))

    if st.button("🤖 Explain Prediction with AI"):
        with st.spinner("AI analyzing prediction..."):
            prompt = f"""A Random Forest model predicts a tree with:
- Species: {species}, Age: {age}yrs, Height: {height}m, Diameter: {diameter}cm
- Soil: {soil}, Location: {barangay}, {municipality}
- Survival Probability: {prob:.1%} → {'ALIVE' if pred else 'DEAD'}

Explain why this tree {'would survive' if pred else 'might die'} in plain language for a DENR field officer. Suggest management actions."""
            resp = ask_ollama(prompt)
            st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

    with st.expander("📈 Advanced Model Performance"):
        tab1, tab2, tab3 = st.tabs(["Metrics", "Cross-Validation", "Feature Importance"])
        with tab1:
            y_pred = get_model().predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, get_model().predict_proba(X_test)[:, 1])
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{acc:.2%}")
            c2.metric("ROC AUC", f"{auc:.3f}")
            st.text(classification_report(y_test, y_pred, target_names=['Dead', 'Alive']))
        with tab2:
            st.metric("Cross-Validation Accuracy (5-Fold)", f"{cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
            st.metric("Logistic Regression (baseline)", f"{accuracy_score(y_test, models["Logistic Regression"].predict(X_test)):.2%}")
        with tab3:
            imp_df = pd.DataFrame({'Feature': perm_imp.importances_mean, 'Name': feature_cols}).sort_values('Feature', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=imp_df, x='Feature', y='Name', palette='viridis', ax=ax)
            ax.set_title('Permutation Feature Importance', fontweight='bold')
            ax.set_xlabel('Importance'); ax.set_ylabel('')
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
#  3. MORTALITY RISK
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Mortality Risk":
    st.markdown('<div class="card"><h3>⚠️ Tree Mortality Risk Classification</h3><p>AI-powered risk assessment with management recommendations.</p></div>', unsafe_allow_html=True)

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
        inp = encode_input(make_input_row(species, 'Libertad', municipality, 8.5, 126.0, age, height, diameter, soil))
        prob_dead = get_model().predict_proba(inp)[0, 0]

        if prob_dead < 0.25:
            color, badge, label, desc = "#2e7d32", "🟢 Low", "Low Mortality Risk", "Tree is likely to survive. No intervention needed."
        elif prob_dead < 0.50:
            color, badge, label, desc = "#f57f17", "🟡 Medium", "Moderate Mortality Risk", "Monitor regularly. Consider preventive measures."
        else:
            color, badge, label, desc = "#c62828", "🔴 High", "High Mortality Risk", "Immediate intervention recommended. Assess site conditions."

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card" style="border-left:4px solid {color}"><div class="label">Mortality Risk</div><div style="font-size:1.4rem;font-weight:700;">{badge} {label}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left:4px solid {color}"><div class="label">Death Probability</div><div style="font-size:1.6rem;font-weight:700;">{prob_dead:.1%}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left:4px solid {color}"><div class="label">Survival Probability</div><div style="font-size:1.6rem;font-weight:700;">{1-prob_dead:.1%}</div></div>', unsafe_allow_html=True)

        if prob_dead < 0.25: st.success(desc)
        elif prob_dead < 0.50: st.warning(desc)
        else: st.error(desc)

        if st.button("🤖 AI Risk Assessment Report"):
            with st.spinner("Generating risk assessment..."):
                prompt = f"""As a forestry expert, assess this tree:
- Species: {species}, Age: {age}yrs, Height: {height}m, Diameter: {diameter}cm
- Soil: {soil}, Municipality: {municipality}
- Death probability: {prob_dead:.1%} — Risk Level: {label}

Provide:
1. Brief risk assessment (2 sentences)
2. Top 3 management recommendations
3. Monitoring schedule suggestion"""
                resp = ask_ollama(prompt)
                st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  4. SPECIES RECOMMENDATION
# ══════════════════════════════════════════════════════════════
elif page == "🌿 Species Recommendation":
    st.markdown('<div class="card"><h3>🌿 AI Species Recommendation Engine</h3><p>Get data-driven and AI-powered species recommendations based on site conditions.</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        soil_type = st.selectbox("Soil Type", df['Soil_Type'].unique(), key='rec_soil')
        municipality = st.selectbox("Municipality", df['Municipality'].unique(), key='rec_mun')
    with col2:
        goal = st.selectbox("Management Goal", ["Maximize Survival", "Maximize Timber", "Maximize Carbon Storage", "Balanced"], key='rec_goal')

    if st.button("Recommend Species", type="primary"):
        sub = df[(df['Soil_Type'] == soil_type) & (df['Municipality'] == municipality)]
        if sub.empty:
            st.warning("No data for this combination. Using soil-type data.")
            sub = df[df['Soil_Type'] == soil_type]

        stats = sub.groupby('Species').agg(
            Count=('Tree_ID', 'count'),
            Survival_Rate=('Survival_Status', lambda x: (x == 'Alive').mean()),
            Avg_Height=('Height_m', 'mean'),
            Avg_Diameter=('Diameter_cm', 'mean'),
            Carbon_Potential=('Height_m', lambda x: (0.05 * (sub.loc[x.index, 'Diameter_cm']**2 * x).mean() * 0.47))
        ).sort_values('Survival_Rate', ascending=False)

        # Score based on goal
        if goal == "Maximize Survival":
            stats['Score'] = stats['Survival_Rate']
            best = stats['Score'].idxmax()
        elif goal == "Maximize Timber":
            stats['Score'] = stats['Avg_Height'] * stats['Avg_Diameter'] * stats['Survival_Rate']
            best = stats['Score'].idxmax()
        elif goal == "Maximize Carbon Storage":
            stats['Score'] = stats['Carbon_Potential'] * stats['Survival_Rate']
            best = stats['Score'].idxmax()
        else:
            stats['Score'] = (stats['Survival_Rate'] * 0.4 + (stats['Avg_Height']/stats['Avg_Height'].max()) * 0.3 + (stats['Avg_Diameter']/stats['Avg_Diameter'].max()) * 0.3)
            best = stats['Score'].idxmax()

        st.markdown(f'<div class="rec-box"><strong>🏆 Top Recommendation: {best}</strong><br>Goal: {goal} | Soil: {soil_type} | Score: {stats.loc[best, "Score"]:.3f}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Survival Rate", f"{stats.loc[best, 'Survival_Rate']:.1%}")
        c2.metric("Avg Height", f"{stats.loc[best, 'Avg_Height']:.1f} m")
        c3.metric("Avg Diameter", f"{stats.loc[best, 'Avg_Diameter']:.1f} cm")
        c4.metric("Trees Recorded", int(stats.loc[best, 'Count']))

        st.dataframe(stats.style.format({
            'Survival_Rate': '{:.1%}', 'Avg_Height': '{:.1f} m', 'Avg_Diameter': '{:.1f} cm',
            'Carbon_Potential': '{:.3f} t', 'Score': '{:.3f}'
        }), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ['#2e7d32' if s == best else '#adb5bd' for s in stats.index]
        sns.barplot(data=stats.reset_index(), x='Survival_Rate', y='Species', palette=colors, ax=ax)
        ax.set_title(f'Species Survival Rate — {soil_type} Soil ({goal})', fontweight='bold')
        ax.set_xlabel('Survival Rate'); ax.set_ylabel('')
        st.pyplot(fig)

        if st.button("🤖 AI Recommendation with Reasoning"):
            with st.spinner("AI generating recommendation..."):
                top3 = stats.head(3).index.tolist()
                prompt = f"""As a forestry expert, recommend the best species for:
- Soil: {soil_type}, Municipality: {municipality}, Goal: {goal}
- Top candidates: {', '.join(top3)}
- Best pick: {best} ({stats.loc[best, 'Survival_Rate']:.1%} survival)

Explain why {best} is the best choice, its ecological benefits, and planting recommendations."""
                resp = ask_ollama(prompt)
                st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  5. GROWTH PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "📈 Growth Prediction":
    st.markdown('<div class="card"><h3>📈 Tree Growth Prediction</h3><p>Predict future growth with AI-powered trend analysis.</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        species = st.selectbox("Species", df['Species'].unique(), key='gr_species')
    with col2:
        soil = st.selectbox("Soil Type", df['Soil_Type'].unique(), key='gr_soil')
    with col3:
        current_age = st.slider("Current Age (Years)", 1, 40, 5, key='gr_age')
    future_age = st.slider("Predict at Age (Years)", current_age + 1, 50, current_age + 5, key='gr_fage')

    if st.button("Predict Growth", type="primary"):
        sub = df[(df['Species'] == species) & (df['Soil_Type'] == soil)]
        if len(sub) < 5:
            sub = df[df['Species'] == species]
            st.info("Limited data; using species-level trends.")
        sub = sub.copy()
        le_g = LabelEncoder()
        sub['Soil_Type_E'] = le_g.fit_transform(sub['Soil_Type'])
        Xg = sub[['Age_Years', 'Soil_Type_E']]
        rf_h = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_d = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_h.fit(Xg, sub['Height_m']); rf_d.fit(Xg, sub['Diameter_cm'])

        inp_g = pd.DataFrame([[future_age, le_g.transform([soil])[0]]], columns=['Age_Years', 'Soil_Type_E'])
        inp_c = pd.DataFrame([[current_age, le_g.transform([soil])[0]]], columns=['Age_Years', 'Soil_Type_E'])
        pred_h, pred_d = rf_h.predict(inp_g)[0], rf_d.predict(inp_g)[0]
        cur_h, cur_d = rf_h.predict(inp_c)[0], rf_d.predict(inp_c)[0]

        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card" style="border-left:4px solid #2e7d32"><div class="label">Height at Age {future_age}</div><div style="font-size:1.6rem;font-weight:700;">{pred_h:.1f} m</div><div style="color:#2e7d32;">▲ {pred_h-cur_h:.1f} m growth</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left:4px solid #5d4037"><div class="label">Diameter at Age {future_age}</div><div style="font-size:1.6rem;font-weight:700;">{pred_d:.1f} cm</div><div style="color:#5d4037;">▲ {pred_d-cur_d:.1f} cm growth</div></div>', unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        age_range = np.arange(1, 51)
        inp_range = pd.DataFrame({'Age_Years': age_range, 'Soil_Type_E': le_g.transform([soil])[0]})
        h_preds = rf_h.predict(inp_range); d_preds = rf_d.predict(inp_range)
        for ax, data, color, ylabel in [
            (axes[0], h_preds, '#2e7d32', 'Height (m)'),
            (axes[1], d_preds, '#5d4037', 'Diameter (cm)')
        ]:
            ax.plot(age_range, data, color=color, linewidth=2.5)
            ax.axvline(current_age, ls='--', color='gray', alpha=0.6)
            ax.axvline(future_age, ls='--', color='orange', alpha=0.6)
            ax.fill_between(age_range, data, alpha=0.08, color=color)
            ax.set_xlabel('Age (Years)'); ax.set_ylabel(ylabel)
        axes[0].set_title(f'{species} — Height Growth', fontweight='bold')
        axes[1].set_title(f'{species} — Diameter Growth', fontweight='bold')
        st.pyplot(fig)

        if st.button("🤖 AI Growth Analysis"):
            with st.spinner("Analyzing growth patterns..."):
                prompt = f"""As a forestry expert, analyze this growth projection:
- Species: {species}, Soil: {soil}
- Current: Age {current_age}yrs, Height {cur_h:.1f}m, Diameter {cur_d:.1f}cm
- Projected at {future_age}yrs: Height {pred_h:.1f}m, Diameter {pred_d:.1f}cm
- Growth rate: {(pred_h-cur_h)/(future_age-current_age):.2f} m/yr height, {(pred_d-cur_d)/(future_age-current_age):.2f} cm/yr diameter

Provide: 1) Growth assessment 2) Comparison to expected rates 3) Management advice"""
                resp = ask_ollama(prompt)
                st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  6. CARBON STORAGE
# ══════════════════════════════════════════════════════════════
elif page == "🌳 Carbon Storage":
    st.markdown('<div class="card"><h3>🌳 Carbon Storage Estimation</h3><p>Estimate carbon sequestration and assess carbon credit potential.</p></div>', unsafe_allow_html=True)

    @st.cache_data
    def fit_carbon_model():
        sub = df.copy(); sub['D2H'] = sub['Diameter_cm']**2 * sub['Height_m']
        le_c = LabelEncoder(); sub['Species_E'] = le_c.fit_transform(sub['Species'])
        rf_c = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_c.fit(sub[['D2H', 'Species_E']], 0.05 * sub['D2H'] * 0.47)
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
        d2h_val = diameter**2 * height
        inp_c = pd.DataFrame([[d2h_val, carb_le.transform([species])[0]]], columns=['D2H', 'Species_E'])
        carbon_t = rf_carb.predict(inp_c)[0]
        co2_eq = carbon_t * 3.67
        trees_for_1t = max(1, int(1 / max(carbon_t, 0.001)))

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card" style="border-left:4px solid #2e7d32"><div class="label">Above-Ground Carbon</div><div style="font-size:1.6rem;font-weight:700;">{carbon_t:.3f} t</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left:4px solid #1565c0"><div class="label">CO₂ Equivalent</div><div style="font-size:1.6rem;font-weight:700;">{co2_eq:.3f} t</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left:4px solid #f57f17"><div class="label">Trees per Tonne CO₂</div><div style="font-size:1.6rem;font-weight:700;">~{trees_for_1t}</div></div>', unsafe_allow_html=True)

        with st.expander("Carbon Credit Potential"):
            credit_price = 15  # USD per credit
            credits = carbon_t * 0.6  # 60% recoverable
            st.markdown(f"""
            **Estimated Carbon Credits:** {credits:.4f} credits (at 60% recovery rate)  
            **Potential Value:** ${credits * credit_price:.2f} USD (at ${credit_price}/credit)  
            *Note: Actual credit values depend on certification standards and market conditions.*
            """)

        if st.button("🤖 AI Carbon Assessment"):
            with st.spinner("AI analyzing carbon potential..."):
                total_forest_c = (0.05 * (df['Diameter_cm']**2 * df['Height_m']).sum() * 0.47) / 1000
                prompt = f"""As a carbon forestry expert, assess this tree's carbon potential:
- Species: {species}, Diameter: {diameter}cm, Height: {height}m
- Carbon: {carbon_t:.3f}t, CO₂e: {co2_eq:.3f}t
- Forest total: {total_forest_c:.1f}t CO₂e across {len(df)} trees

Provide: 1) Carbon assessment 2) Sequestration potential 3) Recommendations for maximizing carbon storage."""
                resp = ask_ollama(prompt)
                st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  7. TIMBER VOLUME
# ══════════════════════════════════════════════════════════════
elif page == "📦 Timber Volume":
    st.markdown('<div class="card"><h3>📦 Timber Volume Prediction</h3><p>Estimates merchantable timber volume with market insights.</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        species = st.selectbox("Species", df['Species'].unique(), key='tv_sp')
    with col2:
        diameter = st.slider("Diameter at Breast Height (cm)", 5.0, 60.0, 25.0, key='tv_d')
    with col3:
        height = st.slider("Height (m)", 2.0, 40.0, 15.0, key='tv_h')

    if st.button("Predict Volume", type="primary"):
        ff = 0.45
        vol_m3 = 0.00007854 * (diameter**2) * height * ff
        bd_ft = vol_m3 * 423.776

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card" style="border-left:4px solid #5d4037"><div class="label">Volume</div><div style="font-size:1.6rem;font-weight:700;">{vol_m3:.3f} m³</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left:4px solid #f57f17"><div class="label">Volume</div><div style="font-size:1.6rem;font-weight:700;">{bd_ft:,.0f} bd ft</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left:4px solid #1976d2"><div class="label">Form Factor</div><div style="font-size:1.6rem;font-weight:700;">{ff}</div></div>', unsafe_allow_html=True)

        if st.button("🤖 AI Market Insight"):
            with st.spinner("Gathering market insights..."):
                prompt = f"""As a timber industry expert, provide market insights for:
- Species: {species}, Volume: {vol_m3:.3f}m³ ({bd_ft:,.0f} bd ft)
- Common uses for {species} in the Philippines
- Market demand and estimated value
- Sustainable harvesting recommendations"""
                resp = ask_ollama(prompt)
                st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  8. GIS PRIORITY MAPPING
# ══════════════════════════════════════════════════════════════
elif page == "🗺️ GIS Priority Mapping":
    st.markdown('<div class="card"><h3>🗺️ GIS-Based Reforestation Priority Mapping</h3><p>AI-enhanced priority area identification with intervention recommendations.</p></div>', unsafe_allow_html=True)

    agg = df.groupby(['Municipality', 'Barangay', 'Soil_Type']).agg(
        Tree_Count=('Tree_ID', 'count'),
        Mortality_Rate=('Survival_Status', lambda x: (x == 'Dead').mean()),
        Avg_Age=('Age_Years', 'mean'), Avg_Height=('Height_m', 'mean')
    ).reset_index()
    agg['Priority_Score'] = (agg['Mortality_Rate'] * 0.4 + (1 - agg['Avg_Height']/df['Height_m'].max()) * 0.3 + (1 - agg['Avg_Age']/df['Age_Years'].max()) * 0.3)
    agg['Priority'] = pd.cut(agg['Priority_Score'], bins=[0, 0.33, 0.66, 1.0], labels=['🟢 Low', '🟡 Medium', '🔴 High'])

    def color_priority(val):
        if '🔴' in str(val): return 'background-color: #ffebee'
        if '🟡' in str(val): return 'background-color: #fff8e1'
        if '🟢' in str(val): return 'background-color: #e8f5e9'
        return ''
    st.dataframe(agg.sort_values('Priority_Score', ascending=False).style.map(color_priority, subset=['Priority']), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Priority Areas")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        top10 = agg.sort_values('Priority_Score', ascending=False).head(10)
        bars = sns.barplot(data=top10, x='Priority_Score', y='Barangay', hue='Municipality', ax=ax, dodge=False)
        ax.set_xlabel('Priority Score'); ax.set_ylabel(''); ax.set_title('Top 10 Barangays', fontweight='bold'); ax.legend(loc='lower right')
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Priority Distribution")
        priority_counts = agg['Priority'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#f1c40f', '#e74c3c'], startangle=90, explode=[0.02]*3, textprops={'fontweight': 'bold'})
        ax.set_title('Priority Level Distribution', fontweight='bold')
        st.pyplot(fig); st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🤖 AI Priority Analysis & Recommendations"):
        with st.spinner("Generating priority analysis..."):
            high_areas = agg[agg['Priority'] == '🔴 High'].head(5)
            areas_str = high_areas[['Municipality', 'Barangay', 'Mortality_Rate']].to_string(index=False)
            prompt = f"""As a DENR reforestation planner, analyze these high-priority areas:
{areas_str}

Provide: 1) Summary of critical areas 2) Recommended interventions per area 3) Resource allocation strategy 4) Timeline recommendations"""
            resp = ask_ollama(prompt)
            st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  9. AI ASSISTANT
# ══════════════════════════════════════════════════════════════
elif page == "🤖 AI Assistant":
    st.markdown('<div class="card"><h3>🤖 AI Forestry Assistant</h3><p>Ask anything about the forest inventory data, get AI-powered insights and recommendations.</p></div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📋 Quick Queries", "🎯 Recommendations", "📊 Data Analysis"])

    # ── Chat Tab ──
    with tab1:
        for msg in st.session_state.chat_history:
            cls = "chat-user" if msg["role"] == "user" else "chat-ai"
            st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

        query = st.chat_input("Ask about the forest data...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    data_context = f"""Dataset: {len(df)} trees, {df['Species'].nunique()} species, {df['Municipality'].nunique()} municipalities.
Survival rate: {(df['Survival_Status']=='Alive').mean():.1%}. Soil types: {df['Soil_Type'].unique().tolist()}.
Species: {df['Species'].unique().tolist()}. Columns: {', '.join(df.columns)}."""
                    prompt = f"""You are a DENR forestry AI assistant. Use this data context:
{data_context}

User question: {query}

Answer concisely and helpfully. Include data-driven insights and actionable recommendations where relevant."""
                    response = ""
                    for chunk in stream_ollama(prompt):
                        response += chunk
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

    # ── Quick Queries ──
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌲 Best performing species"):
                with st.spinner(): 
                    best = df.groupby('Species')['Survival_Status'].apply(lambda x: (x=='Alive').mean()).idxmax()
                    r = ask_ollama(f"Why is {best} the most resilient species? List top 3 ecological traits. Keep it brief.")
                    st.markdown(f'<div class="ai-box"><strong>Best Species: {best}</strong><br>{r}</div>', unsafe_allow_html=True)
            if st.button("⚠️ Most critical municipality"):
                with st.spinner():
                    worst = df.groupby('Municipality')['Survival_Status'].apply(lambda x: (x=='Alive').mean()).idxmin()
                    r = ask_ollama(f"What interventions would help {worst} municipality improve forest survival? 3 recommendations.")
                    st.markdown(f'<div class="ai-box"><strong>Critical: {worst}</strong><br>{r}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("🌍 Carbon summary"):
                with st.spinner():
                    total_c = 0.05 * (df['Diameter_cm']**2 * df['Height_m']).sum() * 0.47 / 1000
                    r = ask_ollama(f"The forest stores {total_c:.1f}t CO₂e across {len(df)} trees. Give a brief carbon impact statement.")
                    st.markdown(f'<div class="ai-box"><strong>Carbon: {total_c:.1f}t CO₂e</strong><br>{r}</div>', unsafe_allow_html=True)
            if st.button("📈 Reforestation strategy"):
                with st.spinner():
                    r = ask_ollama(f"Forest survival rate is {(df['Survival_Status']=='Alive').mean():.1%}. Suggest a 3-point reforestation strategy for DENR.")
                    st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)

    # ── Recommendations Tab ──
    with tab3:
        st.subheader("🎯 Smart Recommendations")
        rec_type = st.selectbox("Recommendation Type", [
            "Species Selection Guide", "Site-Specific Management Plan",
            "Reforestation Strategy", "Intervention Prioritization",
            "Carbon Credit Potential", "Comprehensive Forest Management Plan"
        ])

        if st.button("Generate Recommendation", type="primary"):
            with st.spinner("AI generating recommendation..."):
                prompts = {
                    "Species Selection Guide": f"""Create a species selection guide for Philippine reforestation.
Available species: {df['Species'].unique().tolist()}
Soil types: {df['Soil_Type'].unique().tolist()}
Municipalities: {df['Municipality'].unique().tolist()}
Survival rates by species: {df.groupby('Species')['Survival_Status'].apply(lambda x: (x=='Alive').mean()).to_dict()}

Provide a practical guide: best species per soil type, survival rates, growth characteristics, and planting recommendations.""",
                    "Site-Specific Management Plan": f"""Create a site-specific forest management plan.
Overall survival: {(df['Survival_Status']=='Alive').mean():.1%}
Best area: {df.groupby('Municipality')['Survival_Status'].apply(lambda x: (x=='Alive').mean()).idxmax()}
Worst area: {df.groupby('Municipality')['Survival_Status'].apply(lambda x: (x=='Alive').mean()).idxmin()}

Provide management zones, recommended actions per zone, and monitoring schedule.""",
                    "Reforestation Strategy": f"""Design a reforestation strategy for DENR.
Total area covers {df['Municipality'].nunique()} municipalities
Current survival: {(df['Survival_Status']=='Alive').mean():.1%}
Priority areas with mortality >35%: {df.groupby('Municipality')['Survival_Status'].apply(lambda x: (x=='Dead').mean()).to_dict()}

Provide a phased reforestation plan with species selection, timeline, and success metrics.""",
                    "Intervention Prioritization": f"""Prioritize interventions across municipalities.
Data: {df.groupby('Municipality').agg(Total=('Tree_ID','count'), Mortality=('Survival_Status',lambda x: (x=='Dead').mean())).to_string()}

Rank municipalities by intervention urgency and recommend specific actions for each.""",
                    "Carbon Credit Potential": f"""Assess carbon credit potential.
Forest stats: {len(df)} trees, avg DBH {df['Diameter_cm'].mean():.1f}cm, avg height {df['Height_m'].mean():.1f}m
Total estimated carbon: {0.05 * (df['Diameter_cm']**2 * df['Height_m']).sum() * 0.47 / 1000:.1f}t CO₂e

Provide carbon credit feasibility assessment, methodology recommendation, and potential value.""",
                    "Comprehensive Forest Management Plan": f"""Create a comprehensive forest management plan.
Dataset: {len(df)} trees, {df['Species'].nunique()} species, {df['Municipality'].nunique()} municipalities
Survival: {(df['Survival_Status']=='Alive').mean():.1%}
Soil types: {df['Soil_Type'].unique().tolist()}
Species survival: {df.groupby('Species')['Survival_Status'].apply(lambda x: (x=='Alive').mean()).to_dict()}
Municipality survival: {df.groupby('Municipality')['Survival_Status'].apply(lambda x: (x=='Alive').mean()).to_dict()}

Provide: 1) Executive Summary 2) Situation Analysis 3) Recommended Interventions 4) Species Selection 5) Timeline 6) Budget Considerations 7) Success Metrics"""
                }
                resp = ask_ollama(prompts[rec_type], model="qwen3:4b")
                st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

    # ── Data Analysis Tab ──
    with tab4:
        st.subheader("📊 AI Data Analyst")
        analysis_type = st.selectbox("Analysis Type", [
            "General Dataset Summary", "Species Performance Analysis",
            "Soil Type Effectiveness", "Age Structure Analysis",
            "Location-Based Trends", "Predictive Modeling Insights"
        ])
        if st.button("Run Analysis", type="primary", key="analysis_btn"):
            with st.spinner("AI analyzing data..."):
                prompts = {
                    "General Dataset Summary": f"Analyze this forest inventory dataset: {df.describe().to_string()}. Total: {len(df)} trees, {df['Species'].nunique()} species, {df['Municipality'].nunique()} municipalities. Survival: {(df['Survival_Status']=='Alive').mean():.1%}. Provide key findings and trends.",
                    "Species Performance Analysis": f"Analyze species performance: {df.groupby('Species').agg(Count=('Tree_ID','count'), Survival=('Survival_Status',lambda x: (x=='Alive').mean()), AvgH=('Height_m','mean')).to_string()}. Rank species and provide insights.",
                    "Soil Type Effectiveness": f"Analyze soil type effectiveness: {df.groupby('Soil_Type').agg(Count=('Tree_ID','count'), Survival=('Survival_Status',lambda x: (x=='Alive').mean())).to_string()}. Which soil types perform best and why?",
                    "Age Structure Analysis": f"Analyze age structure: Young (<=5yrs): {(df[df['Age_Years']<=5]['Survival_Status']=='Alive').mean():.1%} survival, Mature (>20yrs): {(df[df['Age_Years']>20]['Survival_Status']=='Alive').mean() if len(df[df['Age_Years']>20])>0 else 0:.1%}. Provide age management recommendations.",
                    "Location-Based Trends": f"Analyze location-based trends: {df.groupby('Municipality').agg(Count=('Tree_ID','count'), Survival=('Survival_Status',lambda x: (x=='Alive').mean()), AvgAge=('Age_Years','mean')).to_string()}. Identify geographical patterns.",
                    "Predictive Modeling Insights": f"Model ({st.session_state.model_choice}) performance: Accuracy: {accuracy_score(y_test, get_model().predict(X_test)):.2%}, CV: {cv_scores.mean():.2%}±{cv_scores.std():.2%}. Key features: {', '.join(feature_cols[:5])}. Provide insights on model reliability and key drivers."
                }
                resp = ask_ollama(prompts[analysis_type], model="qwen3:4b")
                st.markdown(f'<div class="ai-box">{resp}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  10. NEWS REPORT SUMMARY
# ══════════════════════════════════════════════════════════════
elif page == "📰 News Report Summary":
    total = len(df); alive = (df['Survival_Status'] == 'Alive').sum(); dead = (df['Survival_Status'] == 'Dead').sum(); survival_rate = alive/total

    mun_stats = df.groupby('Municipality').agg(Total_Trees=('Tree_ID','count'), Alive=('Survival_Status',lambda x:(x=='Alive').sum()), Dead=('Survival_Status',lambda x:(x=='Dead').sum()), Survival_Rate=('Survival_Status',lambda x:(x=='Alive').mean()), Avg_Age=('Age_Years','mean'), Avg_Height=('Height_m','mean')).reset_index().sort_values('Survival_Rate')
    best_mun = mun_stats.loc[mun_stats['Survival_Rate'].idxmax()]; worst_mun = mun_stats.loc[mun_stats['Survival_Rate'].idxmin()]
    sp_stats = df.groupby('Species').agg(Count=('Tree_ID','count'), Survival_Rate=('Survival_Status',lambda x:(x=='Alive').mean()), Avg_Height=('Height_m','mean'), Avg_Diameter=('Diameter_cm','mean'))
    best_sp = sp_stats['Survival_Rate'].idxmax(); worst_sp = sp_stats['Survival_Rate'].idxmin()
    soil_stats = df.groupby('Soil_Type').agg(Survival_Rate=('Survival_Status',lambda x:(x=='Alive').mean()), Count=('Tree_ID','count'))
    best_soil = soil_stats['Survival_Rate'].idxmax()
    young_trees = df[df['Age_Years']<=5]; mature_trees = df[df['Age_Years']>20]
    young_survival = (young_trees['Survival_Status']=='Alive').mean()
    mature_survival = (mature_trees['Survival_Status']=='Alive').mean() if len(mature_trees)>0 else 0
    total_carbon = 0.05 * (df['Diameter_cm']**2 * df['Height_m']).sum() * 0.47 / 1000

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Forest Overview")
    cols = st.columns(4)
    cols[0].metric("Total Trees", total); cols[1].metric("Alive", f"{alive} ({survival_rate:.1%})")
    cols[2].metric("Dead", f"{dead} ({1-survival_rate:.1%})"); cols[3].metric("Municipalities", df['Municipality'].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

    status_color, status_icon, status_label = ("#2e7d32","✅","Healthy") if survival_rate>=0.70 else ("#f57f17","⚠️","Moderate Concern") if survival_rate>=0.50 else ("#c62828","🔴","Critical")
    st.markdown(f'<div class="card" style="border-left:5px solid {status_color};background:{status_color}08;"><div style="display:flex;align-items:center;gap:12px;"><span style="font-size:2rem;">{status_icon}</span><div><strong style="color:{status_color};font-size:1.1rem;">Forest Status: {status_label}</strong><p style="margin:2px 0 0;color:#666;">{survival_rate:.1%} overall survival rate</p></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Key Insights</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**🏆 Best Municipality:** {best_mun['Municipality']}"); st.caption(f"{best_mun['Survival_Rate']:.1%} survival")
        st.markdown(f"**🌿 Best Species:** {best_sp}"); st.caption(f"{sp_stats.loc[best_sp,'Survival_Rate']:.1%} survival")
    with c2:
        st.markdown(f"**⚠️ Needs Intervention:** {worst_mun['Municipality']}"); st.caption(f"{worst_mun['Survival_Rate']:.1%} survival")
        st.markdown(f"**🌍 Carbon Stored:** {total_carbon:.1f}t CO₂e")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Alerts</h3>', unsafe_allow_html=True)
    has_alert = False
    for _, m in mun_stats.iterrows():
        if m['Survival_Rate'] < 0.50: st.error(f"**{m['Municipality']}** — Critical ({m['Survival_Rate']:.1%}). Immediate action needed."); has_alert = True
        elif m['Survival_Rate'] < 0.65: st.warning(f"**{m['Municipality']}** — Moderate ({m['Survival_Rate']:.1%}). Monitor."); has_alert = True
    if not has_alert: st.success("All municipalities have adequate survival rates.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Per-Municipality Report"):
        st.dataframe(mun_stats.style.format({'Survival_Rate':'{:.1%}','Avg_Age':'{:.1f} yrs','Avg_Height':'{:.1f} m'}), use_container_width=True)

    st.markdown('<div class="card"><h3>Live Forestry News</h3>', unsafe_allow_html=True)
    st.caption("Sourced from Google News — forestry, environment, and DENR updates")
    @st.cache_data(ttl=600)
    def fetch_live_news():
        sources = [
            "https://news.google.com/rss/search?q=forestry+Philippines+DENR&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=reforestation+environment+Philippines&hl=en-PH&gl=PH&ceid=PH:en",
        ]
        seen, entries = set(), []
        for url in sources:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:5]:
                    t = e.title.strip()
                    if t not in seen:
                        seen.add(t)
                        image_url = None
                        if hasattr(e,'media_content'):
                            for mc in e.media_content:
                                if mc.get('type','').startswith('image'): image_url=mc['url']; break
                        if not image_url and hasattr(e,'summary'):
                            m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', e.summary)
                            if m: image_url=m.group(1)
                        entries.append({
                            "title": t, "link": e.link,
                            "source": getattr(e,"source",{}).get("title","Google News") if hasattr(e,"source") else "Google News",
                            "published": getattr(e,"published","Just now"),
                            "summary": re.sub(r'<[^>]+>','',getattr(e,'summary',''))[:200],
                            "image_url": image_url
                        })
            except: pass
            time.sleep(0.3)
        return entries[:10]
    if st.button("Refresh Live News", key="refresh_news"): st.cache_data.clear(); st.rerun()
    for ne in fetch_live_news():
        with st.container(border=True):
            cols = st.columns([1,3])
            with cols[0]:
                if ne.get('image_url'): st.image(ne['image_url'], use_container_width=True)
            with cols[1]:
                st.markdown(f"**[{ne['title']}]({ne['link']})**")
                st.caption(f"{ne['source']} • {ne['published']}")
                if ne['summary']: st.markdown(ne['summary'])
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Report generated from forest inventory data • For DENR decision support")

# ── SIDEBAR FOOTER ──
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding:0.8rem;background:#f8f9fa;border-radius:8px;font-size:0.75rem;color:#6c757d;">
    <strong>🌳 Forest DSS v2.0</strong><br>
    AI-powered • Ollama • RandomForest • XGBoost-ready<br>
    <span style="opacity:0.6;">For DENR decision support</span>
</div>
""", unsafe_allow_html=True)

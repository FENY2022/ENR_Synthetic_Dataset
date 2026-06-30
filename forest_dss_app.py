import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

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

DATA_PATH = 'forest_inventory_dataset_1000 - forest_inventory_dataset_1000.csv'

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

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ['#2e7d32' if i == 0 else '#adb5bd' for i in range(len(stats))]
        sns.barplot(data=stats.reset_index(), x='Survival_Rate', y='Species',
                    palette=colors, ax=ax)
        ax.set_title(f'Species Survival Rate — {soil_type} Soil', fontweight='bold')
        ax.set_xlabel('Survival Rate')
        ax.set_ylabel('')
        st.pyplot(fig)

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

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top Priority Areas")
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
            "https://news.google.com/rss/search?q=forestry+Philippines+DENR&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=reforestation+environment+Philippines&hl=en-PH&gl=PH&ceid=PH:en",
            "https://news.google.com/rss/search?q=climate+change+forest+conservation+Philippines&hl=en-PH&gl=PH&ceid=PH:en",
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

    st.caption("Report generated from forest inventory data • For DENR decision support")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding:0.8rem; background:#f8f9fa; border-radius:8px; font-size:0.75rem; color:#6c757d;">
    <strong>About</strong><br>
    Built with Streamlit • Random Forest models trained on forest inventory data.
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

st.set_page_config(page_title="Forest DSS", layout="wide")
st.title("🌳 DENR Forest Decision Support System")
st.markdown("---")

DATA_PATH = 'forest_inventory_dataset_1000 - forest_inventory_dataset_1000.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# ── Sidebar ──
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Data Overview",
    "🌱 Survival Prediction",
    "⚠️ Mortality Risk",
    "🌿 Species Recommendation",
    "📈 Growth Prediction",
    "🌳 Carbon Storage",
    "📦 Timber Volume",
    "🗺️ GIS Priority Mapping",
    "📰 News Report Summary",
])

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

# ── 1. DATA OVERVIEW ──
if page == "📊 Data Overview":
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trees", df.shape[0])
    col2.metric("Species", df['Species'].nunique())
    col3.metric("Alive", (df['Survival_Status'] == 'Alive').sum())
    col4.metric("Dead", (df['Survival_Status'] == 'Dead').sum())

    st.dataframe(df.head(20), width='stretch')
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), width='stretch')

    st.subheader("Survival by Species")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, y='Species', hue='Survival_Status', ax=ax)
    st.pyplot(fig)

    st.subheader("Survival by Soil Type")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='Soil_Type', hue='Survival_Status', ax=ax2)
    st.pyplot(fig2)

# ── 2. SURVIVAL PREDICTION ──
elif page == "🌱 Survival Prediction":
    st.subheader("Tree Survival Prediction")
    st.markdown("Enter tree details below to predict whether the tree will survive.")

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
        cola, colb = st.columns(2)
        cola.metric("Prediction", "✅ Alive" if pred == 1 else "❌ Dead")
        colb.metric("Survival Probability", f"{prob:.2%}")
        st.progress(float(prob))

    st.markdown("---")
    st.subheader("Model Performance")
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    cola, colb = st.columns(2)
    cola.metric("Accuracy", f"{acc:.2%}")
    colb.metric("ROC AUC", f"{auc:.3f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=['Dead', 'Alive']))

# ── 3. MORTALITY RISK ──
elif page == "⚠️ Mortality Risk":
    st.subheader("Tree Mortality Risk Classification")
    st.markdown("Classifies trees into **Low**, **Medium**, or **High** mortality risk.")

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
        if prob_dead < 0.25:
            risk = "🟢 Low"
            desc = "Tree is likely to survive. No immediate intervention needed."
        elif prob_dead < 0.50:
            risk = "🟡 Medium"
            desc = "Tree faces moderate mortality risk. Monitor regularly."
        else:
            risk = "🔴 High"
            desc = "Tree is at high risk of mortality. Consider intervention."

        st.markdown("---")
        cola, colb, colc = st.columns(3)
        cola.metric("Mortality Risk", risk)
        colb.metric("Death Probability", f"{prob_dead:.2%}")
        colc.metric("Survival Probability", f"{1-prob_dead:.2%}")
        st.info(desc)

# ── 4. SPECIES RECOMMENDATION ──
elif page == "🌿 Species Recommendation":
    st.subheader("Species Recommendation Engine")
    st.markdown("Recommends the best tree species based on soil type and location.")

    soil_type = st.selectbox("Soil Type", df['Soil_Type'].unique(), key='rec_soil')
    municipality = st.selectbox("Municipality", df['Municipality'].unique(), key='rec_mun')

    if st.button("Recommend Species", type="primary"):
        sub = df[(df['Soil_Type'] == soil_type) & (df['Municipality'] == municipality)]
        if sub.empty:
            st.warning("No data available for this combination. Showing overall best performers.")
            sub = df[df['Soil_Type'] == soil_type]

        stats = sub.groupby('Species').agg(
            Count=('Tree_ID', 'count'),
            Survival_Rate=('Survival_Status', lambda x: (x == 'Alive').mean()),
            Avg_Height=('Height_m', 'mean'),
            Avg_Diameter=('Diameter_cm', 'mean')
        ).sort_values('Survival_Rate', ascending=False)

        st.dataframe(stats.style.format({
            'Survival_Rate': '{:.1%}',
            'Avg_Height': '{:.1f} m',
            'Avg_Diameter': '{:.1f} cm'
        }), width='stretch')

        best = stats.index[0]
        st.success(f"**Recommended Species: {best}** — {stats.loc[best, 'Survival_Rate']:.1%} survival rate in {soil_type} soil.")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=stats.reset_index(), x='Survival_Rate', y='Species', ax=ax)
        ax.set_title(f'Survival Rate by Species ({soil_type} Soil)')
        ax.set_xlabel('Survival Rate')
        st.pyplot(fig)

# ── 5. GROWTH PREDICTION ──
elif page == "📈 Growth Prediction":
    st.subheader("Tree Growth Prediction")
    st.markdown("Predict future height and diameter based on current age and species.")

    species = st.selectbox("Species", df['Species'].unique(), key='gr_species')
    soil = st.selectbox("Soil Type", df['Soil_Type'].unique(), key='gr_soil')
    current_age = st.slider("Current Age (Years)", 1, 40, 5, key='gr_age')
    future_age = st.slider("Predict at Age (Years)", current_age + 1, 50, current_age + 5, key='gr_fage')

    if st.button("Predict Growth", type="primary"):
        sub = df[(df['Species'] == species) & (df['Soil_Type'] == soil)]
        if len(sub) < 5:
            sub = df[df['Species'] == species]
            st.info("Limited data for this soil type; using species-level data.")

        sub = sub.copy()
        le_g = LabelEncoder()
        sub['Soil_Type_E'] = le_g.fit_transform(sub['Soil_Type'])

        Xg = sub[['Age_Years', 'Soil_Type_E']]
        yh = sub['Height_m']
        yd = sub['Diameter_cm']

        rf_h = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_d = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_h.fit(Xg, yh)
        rf_d.fit(Xg, yd)

        inp_g = pd.DataFrame([[future_age, le_g.transform([soil])[0]]], columns=['Age_Years', 'Soil_Type_E'])
        pred_h = rf_h.predict(inp_g)[0]
        pred_d = rf_d.predict(inp_g)[0]

        # also predict current
        inp_c = pd.DataFrame([[current_age, le_g.transform([soil])[0]]], columns=['Age_Years', 'Soil_Type_E'])
        cur_h = rf_h.predict(inp_c)[0]
        cur_d = rf_d.predict(inp_c)[0]

        col1, col2 = st.columns(2)
        col1.metric(f"Height at Age {future_age}", f"{pred_h:.1f} m", f"{pred_h - cur_h:.1f} m from current")
        col2.metric(f"Diameter at Age {future_age}", f"{pred_d:.1f} cm", f"{pred_d - cur_d:.1f} cm from current")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        age_range = np.arange(1, 51)
        inp_range = pd.DataFrame({'Age_Years': age_range, 'Soil_Type_E': le_g.transform([soil])[0]})
        axes[0].plot(age_range, rf_h.predict(inp_range), color='green')
        axes[0].axvline(current_age, ls='--', color='gray')
        axes[0].axvline(future_age, ls='--', color='orange')
        axes[0].scatter([current_age], [cur_h], color='blue', s=80, zorder=5)
        axes[0].scatter([future_age], [pred_h], color='orange', s=80, zorder=5)
        axes[0].set_xlabel('Age (Years)'); axes[0].set_ylabel('Height (m)')
        axes[0].set_title(f'{species} Height Growth')
        axes[0].legend(['Predicted', 'Current Age', 'Target Age'])

        axes[1].plot(age_range, rf_d.predict(inp_range), color='brown')
        axes[1].axvline(current_age, ls='--', color='gray')
        axes[1].axvline(future_age, ls='--', color='orange')
        axes[1].scatter([current_age], [cur_d], color='blue', s=80, zorder=5)
        axes[1].scatter([future_age], [pred_d], color='orange', s=80, zorder=5)
        axes[1].set_xlabel('Age (Years)'); axes[1].set_ylabel('Diameter (cm)')
        axes[1].set_title(f'{species} Diameter Growth')
        st.pyplot(fig)

# ── 6. CARBON STORAGE ──
elif page == "🌳 Carbon Storage":
    st.subheader("Carbon Storage Estimation")
    st.markdown("Estimates above-ground carbon sequestration based on tree dimensions and species.")

    # Simple allometric model: Biomass = a * (D^2 * H)^b, Carbon = Biomass * 0.47
    @st.cache_data
    def fit_carbon_model():
        sub = df.copy()
        sub['D2H'] = sub['Diameter_cm'] ** 2 * sub['Height_m']
        le_c = LabelEncoder()
        sub['Species_E'] = le_c.fit_transform(sub['Species'])
        Xc = sub[['D2H', 'Species_E']]
        # Assume biomass ~ 0.05 * D^2*H (generic tropical allometric)
        yc = 0.05 * sub['D2H'] * 0.47  # carbon tonnes
        rf_c = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_c.fit(Xc, yc)
        return rf_c, le_c

    rf_carb, carb_le = fit_carbon_model()

    species = st.selectbox("Species", df['Species'].unique(), key='carb_sp')
    diameter = st.slider("Diameter at Breast Height (cm)", 5.0, 60.0, 25.0, key='carb_d')
    height = st.slider("Height (m)", 2.0, 40.0, 15.0, key='carb_h')

    if st.button("Estimate Carbon", type="primary"):
        d2h_val = diameter ** 2 * height
        inp_c = pd.DataFrame([[d2h_val, carb_le.transform([species])[0]]], columns=['D2H', 'Species_E'])
        carbon_t = rf_carb.predict(inp_c)[0]
        co2_eq = carbon_t * 3.67

        col1, col2, col3 = st.columns(3)
        col1.metric("Above-Ground Carbon", f"{carbon_t:.2f} tonnes")
        col2.metric("CO₂ Equivalent", f"{co2_eq:.2f} tonnes")
        col3.metric("Biomass (est.)", f"{carbon_t / 0.47:.2f} tonnes")

        st.markdown("""
        *Estimation based on generic tropical allometric equation:  
        **Biomass = 0.05 × D² × H** where D = diameter (cm), H = height (m).  
        Carbon = Biomass × 0.47, CO₂e = Carbon × 3.67*
        """)

# ── 7. TIMBER VOLUME ──
elif page == "📦 Timber Volume":
    st.subheader("Timber Volume Prediction")
    st.markdown("Predicts merchantable timber volume using tree dimensions.")

    species = st.selectbox("Species", df['Species'].unique(), key='tv_sp')
    diameter = st.slider("Diameter at Breast Height (cm)", 5.0, 60.0, 25.0, key='tv_d')
    height = st.slider("Height (m)", 2.0, 40.0, 15.0, key='tv_h')

    # Volume = 0.00007854 * D^2 * H * F (form factor ~0.45)
    if st.button("Predict Volume", type="primary"):
        form_factor = 0.45
        volume_m3 = 0.00007854 * (diameter ** 2) * height * form_factor
        bd_ft = volume_m3 * 423.776  # cubic meters to board feet

        col1, col2, col3 = st.columns(3)
        col1.metric("Volume (m³)", f"{volume_m3:.3f}")
        col2.metric("Volume (board ft)", f"{bd_ft:.0f}")
        col3.metric("Form Factor", form_factor)

        st.markdown("""
        *Volume = 0.00007854 × D² × H × FF*  
        where D = diameter (cm), H = height (m), FF = form factor (0.45).  
        1 m³ ≈ 423.78 board feet.
        """)

# ── 8. GIS PRIORITY MAPPING ──
elif page == "🗺️ GIS Priority Mapping":
    st.subheader("GIS-Based Reforestation Priority Mapping")
    st.markdown("Identifies high-priority areas for reforestation based on mortality rates and tree density.")

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

    st.dataframe(agg.sort_values('Priority_Score', ascending=False), width='stretch')

    st.subheader("Priority Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=agg.sort_values('Priority_Score', ascending=False).head(10),
                x='Priority_Score', y='Barangay', hue='Municipality', ax=axes[0], dodge=False)
    axes[0].set_title('Top 10 Priority Areas')

    priority_counts = agg['Priority'].value_counts()
    axes[1].pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%',
               colors=['#2ecc71', '#f1c40f', '#e74c3c'])
    axes[1].set_title('Priority Level Breakdown')
    st.pyplot(fig)

    with st.expander("Priority Scoring Methodology"):
        st.markdown("""
        **Priority Score = 0.4 × Mortality Rate + 0.3 × (1 - Normalized Height) + 0.3 × (1 - Normalized Age)**  
        - **High Mortality** → higher priority  
        - **Low average height** → higher priority (young/stunted)  
        - **Low average age** → higher priority (young stands need intervention)
        """)

# ── 9. NEWS REPORT SUMMARY ──
elif page == "📰 News Report Summary":
    # ── CSS Animations ──
    st.markdown("""
    <style>
    @keyframes ticker {
        0%   { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    @keyframes fadeSlideIn {
        0%   { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 5px rgba(255,200,0,0.3); }
        50%      { box-shadow: 0 0 25px rgba(255,200,0,0.8); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.5); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0; }
    }
    .news-ticker-bar {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        color: #fff; padding: 8px 0; border-radius: 6px; overflow: hidden;
        margin-bottom: 20px; border-left: 4px solid #e94560;
    }
    .news-ticker-text {
        display: inline-block; white-space: nowrap;
        animation: ticker 25s linear infinite; padding-left: 100%;
        font-size: 1.1rem; font-weight: 500; letter-spacing: 0.5px;
    }
    .news-ticker-text span { color: #ffd700; margin: 0 20px; }
    .news-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: #fff; padding: 20px; border-radius: 12px; margin: 16px 0;
        animation: fadeSlideIn 0.6s ease-out forwards; opacity: 0;
        border-left: 5px solid #e94560; transition: transform 0.3s;
    }
    .news-card:hover { transform: translateX(8px); }
    .news-card h4 { margin: 0 0 8px 0; color: #ffd700; font-size: 1.15rem; }
    .news-card p  { margin: 0; line-height: 1.6; color: #e0e0e0; }
    .live-badge {
        display: inline-block; background: #e94560; color: #fff;
        padding: 2px 12px; border-radius: 4px; font-size: 0.75rem;
        font-weight: 700; letter-spacing: 1px; margin-right: 10px;
        animation: blink 1.2s infinite;
    }
    .headline-glow {
        animation: pulseGlow 2s infinite; border-radius: 10px; padding: 6px 16px;
    }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460); border-radius: 10px;
        padding: 16px; text-align: center; animation: countUp 0.5s ease-out forwards;
        opacity: 0; border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-box .num { font-size: 1.8rem; font-weight: 700; color: #ffd700; }
    .metric-box .lbl { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    </style>
    """, unsafe_allow_html=True)

    total = len(df)
    alive = (df['Survival_Status'] == 'Alive').sum()
    dead = (df['Survival_Status'] == 'Dead').sum()
    survival_rate = alive / total
    mortality_rate = dead / total

    # ── Compute stats once ──
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

    # ── Breaking News Ticker ──
    ticker_items = [
        f"🌳 {total} trees surveyed across {df['Municipality'].nunique()} municipalities",
        f"✅ Overall survival rate: {survival_rate:.1%}",
        f"🏆 {best_mun['Municipality']} leads with {best_mun['Survival_Rate']:.1%} survival",
        f"⚠️ {worst_mun['Municipality']} needs urgent attention ({worst_mun['Survival_Rate']:.1%})",
        f"🌿 Best species: {best_sp} ({sp_stats.loc[best_sp, 'Survival_Rate']:.1%})",
        f"🧪 Best soil: {best_soil} ({soil_stats.loc[best_soil, 'Survival_Rate']:.1%})",
        f"🌍 Carbon stored: {total_carbon:.1f} tonnes CO₂e",
    ]
    ticker_html = '<div class="news-ticker-bar"><div class="news-ticker-text">'
    for item in ticker_items:
        ticker_html += f'<span>◆</span> {item} '
    ticker_html += '</div></div>'
    st.markdown(ticker_html, unsafe_allow_html=True)

    # ── Live Headline ──
    if survival_rate >= 0.70:
        h_color, h_text = "#2ecc71", "FOREST IN GOOD CONDITION"
        h_detail = f"Overall survival rate is {survival_rate:.1%}, indicating a healthy forest ecosystem across all municipalities."
    elif survival_rate >= 0.50:
        h_color, h_text = "#f1c40f", "MODERATE CONCERN"
        h_detail = f"Survival rate at {survival_rate:.1%}. Some areas need attention and possible intervention."
    else:
        h_color, h_text = "#e94560", "CRITICAL ALERT"
        h_detail = f"Survival rate is only {survival_rate:.1%}. Immediate reforestation and intervention required."

    st.markdown(f"""
    <div style="text-align:center; margin: 10px 0 20px 0;">
        <span class="live-badge">LIVE</span>
        <span style="font-size:0.9rem; color:#888;">Forest News Brief</span>
        <div class="headline-glow" style="background:{h_color}22; border:1px solid {h_color}; margin-top:10px;">
            <h2 style="color:{h_color}; margin:8px 0;">{h_text}</h2>
        </div>
        <p style="color:#ccc; margin-top:8px;">{h_detail}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Animated Metric Counters ──
    st.markdown('<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin:20px 0;">', unsafe_allow_html=True)
    metrics = [
        ("🌳 Total Trees", str(total)),
        ("✅ Survival Rate", f"{survival_rate:.1%}"),
        ("❌ Mortality Rate", f"{mortality_rate:.1%}"),
        ("📈 Health Index", f"{(survival_rate*100):.1f}%"),
    ]
    for i, (lbl, num) in enumerate(metrics):
        st.markdown(f'<div class="metric-box" style="animation-delay:{i*0.1}s"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── News Cards (with staggered animation) ──
    st.markdown('<h3 style="color:#ffd700;">📋 Top Stories</h3>', unsafe_allow_html=True)

    articles = [
        ("🏆 Municipality Leader", f"<b>{best_mun['Municipality']}</b> recorded the highest survival rate at <b>{best_mun['Survival_Rate']:.1%}</b> — {int(best_mun['Alive'])} alive out of {int(best_mun['Total_Trees'])} trees. This area sets the benchmark for reforestation success."),
        ("⚠️ Needs Attention", f"<b>{worst_mun['Municipality']}</b> has the lowest survival rate at <b>{worst_mun['Survival_Rate']:.1%}</b> ({int(worst_mun['Dead'])} dead out of {int(worst_mun['Total_Trees'])} trees). Priority intervention is recommended."),
        ("🌿 Species Spotlight", f"<b>{best_sp}</b> is the most resilient species (<b>{sp_stats.loc[best_sp, 'Survival_Rate']:.1%}</b> survival). <b>{worst_sp}</b> has the lowest rate (<b>{sp_stats.loc[worst_sp, 'Survival_Rate']:.1%}</b>) and may need site-matching adjustments."),
        ("🧪 Soil Insight", f"Trees in <b>{best_soil}</b> soil show the best survival (<b>{soil_stats.loc[best_soil, 'Survival_Rate']:.1%}</b>). This guides future planting site selection."),
        ("📈 Age Structure", f"Young trees (≤5 yrs): <b>{young_survival:.1%}</b> survival ({len(young_trees)} trees). Mature trees (>20 yrs): <b>{mature_survival:.1%}</b> survival ({len(mature_trees)} trees). {'Young trees need extra monitoring.' if young_survival < 0.70 else 'Young trees are establishing well.'}"),
        ("🌍 Carbon Storage", f"Estimated above-ground carbon: <b>{total_carbon:.1f} tonnes</b> — equivalent to <b>{total_carbon * 3.67:.1f} tonnes of CO₂</b> removed from the atmosphere across {total} trees."),
    ]

    for i, (title, body) in enumerate(articles):
        st.markdown(f'<div class="news-card" style="animation-delay:{i*0.12}s"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    # ── AI Live News Feed ──
    st.markdown('<h3 style="color:#ffd700; margin-top:30px;">🤖 AI Live News Feed</h3>', unsafe_allow_html=True)
    st.markdown("Real-time news sourced from Google News RSS — forestry, environment, and DENR updates.")

    @st.cache_data(ttl=600)
    def fetch_live_news():
        import feedparser, time
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
                        entries.append({
                            "title": t,
                            "link": e.link,
                            "source": getattr(e, "source", {}).get("title", "Google News") if hasattr(e, "source") else "Google News",
                            "published": getattr(e, "published", "Just now"),
                            "summary": getattr(e, "summary", "")[:200],
                        })
            except:
                pass
            time.sleep(0.3)
        return entries[:10]

    if st.button("🔄 Refresh Live News", key="refresh_news"):
        st.cache_data.clear()
        st.rerun()

    news_entries = fetch_live_news()
    if news_entries:
        for i, ne in enumerate(news_entries):
            st.markdown(f"""
            <div class="news-card" style="animation-delay:{i*0.08}s; border-left-color:#00b4d8;">
                <span style="font-size:0.7rem; color:#00b4d8; text-transform:uppercase; letter-spacing:1px;">
                    📡 {ne['source']} • {ne['published']}
                </span>
                <h4 style="margin:4px 0 6px 0;">
                    <a href="{ne['link']}" target="_blank" style="color:#ffd700; text-decoration:none;">
                        {ne['title']}
                    </a>
                </h4>
                <p style="font-size:0.9rem;">{ne['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Live news unavailable. Showing data-driven reports below.")

    # ── Per-Municipality Report ──
    with st.expander("🏛️ View Full Per-Municipality Report"):
        st.dataframe(mun_stats.style.format({
            'Survival_Rate': '{:.1%}', 'Avg_Age': '{:.1f} yrs', 'Avg_Height': '{:.1f} m'
        }), width='stretch')

    # ── Alerts ──
    st.markdown('<h3 style="color:#ffd700;">🚨 Live Alerts</h3>', unsafe_allow_html=True)
    has_alert = False
    for _, m in mun_stats.iterrows():
        if m['Survival_Rate'] < 0.50:
            st.error(f"🔴 **{m['Municipality']}** — Critical ({m['Survival_Rate']:.1%}). Immediate reforestation needed.")
            has_alert = True
        elif m['Survival_Rate'] < 0.65:
            st.warning(f"🟡 **{m['Municipality']}** — Moderate ({m['Survival_Rate']:.1%}). Monitoring recommended.")
            has_alert = True
    if not has_alert:
        st.success("✅ No critical alerts. All municipalities have adequate survival rates.")

    st.markdown("---")
    st.caption("📅 Live report generated from forest inventory data • For DENR decision support")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit • Models trained on Forest Inventory Dataset")

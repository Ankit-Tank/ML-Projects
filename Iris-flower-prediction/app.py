import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris · Species Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS — Botanical Luxury Theme ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --cream:      #F5F0E8;
    --parchment:  #EDE5D0;
    --forest:     #1C3A2E;
    --moss:       #2D5440;
    --sage:       #4A7C59;
    --gold:       #B8953A;
    --gold-light: #D4AF5A;
    --blush:      #E8C4B8;
    --lavender:   #C4B8D4;
    --ink:        #1A1A1A;
    --text-muted: #5A5A4A;
    --border:     rgba(184,149,58,0.25);
    --shadow:     0 8px 40px rgba(28,58,46,0.12);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--ink);
}

.main .block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1200px;
}

/* ── Hero Header ── */
.hero-wrapper {
    background: linear-gradient(135deg, var(--forest) 0%, var(--moss) 60%, #3D6B4F 100%);
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(184,149,58,0.18) 0%, transparent 70%);
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -30px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold-light);
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 300;
    color: var(--cream);
    line-height: 1.1;
    margin: 0 0 1rem;
}
.hero-title em {
    font-style: italic;
    color: var(--gold-light);
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(245,240,232,0.72);
    line-height: 1.7;
    max-width: 520px;
    font-weight: 300;
}
.hero-badges {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.8rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.18);
    color: rgba(245,240,232,0.85);
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    font-weight: 400;
}

/* ── Section Titles ── */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.75rem;
    font-weight: 400;
    color: var(--forest);
    margin: 0 0 0.25rem;
}
.section-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    letter-spacing: 0.04em;
}
.gold-rule {
    height: 2px;
    width: 48px;
    background: linear-gradient(90deg, var(--gold), transparent);
    border: none;
    margin: 0.5rem 0 1.5rem;
}

/* ── Cards ── */
.card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 2px 20px rgba(28,58,46,0.06);
    height: 100%;
}
.card-forest {
    background: linear-gradient(135deg, var(--forest), var(--moss));
    border: none;
    color: var(--cream);
}

/* ── Model Selector ── */
div[data-testid="stSelectbox"] > div {
    border-radius: 10px !important;
    border-color: var(--border) !important;
    background: white !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] {
    padding: 0.5rem 0;
}
div[data-testid="stSlider"] > div > div > div {
    background: var(--forest) !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--gold) !important;
    border: 2px solid var(--gold) !important;
}
.slider-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--forest), var(--moss)) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(28,58,46,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(28,58,46,0.4) !important;
}

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, var(--forest) 0%, var(--moss) 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: var(--cream);
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 30px rgba(28,58,46,0.25);
}
.result-box::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 150px; height: 150px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(184,149,58,0.2) 0%, transparent 70%);
}
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; display: block; }
.result-label {
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold-light);
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.result-species {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 400;
    font-style: italic;
    color: white;
    line-height: 1.2;
}
.result-confidence {
    font-size: 0.85rem;
    color: rgba(245,240,232,0.7);
    margin-top: 0.5rem;
}
.confidence-pill {
    display: inline-block;
    background: var(--gold);
    color: var(--forest);
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.2rem 0.8rem;
    border-radius: 100px;
    margin-top: 0.4rem;
}

/* ── Metrics ── */
.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(28,58,46,0.05);
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--forest);
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Species Info Cards ── */
.species-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.species-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(28,58,46,0.12);
}
.species-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.species-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-style: italic;
    color: var(--forest);
    font-weight: 500;
}
.species-desc {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* ── DataFrame styling ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* ── Tabs ── */
div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    color: var(--text-muted) !important;
    font-weight: 400;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--forest) !important;
    font-weight: 500;
}
div[data-testid="stTabs"] [role="tabpanel"] {
    padding-top: 1rem;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    margin: 2rem 0 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
}

/* ── Notification / Info ── */
div[data-testid="stInfo"], div[data-testid="stSuccess"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load & Train Models ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "Iris.csv"))

    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])

    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df[features]
    Y = df["Species"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr  = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, Y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, Y_train)

    nb  = GaussianNB()
    nb.fit(X_train, Y_train)

    models = {"Logistic Regression": lr, "KNN": knn, "Naive Bayes": nb}

    lr_pred  = lr.predict(X_test_scaled)
    knn_pred = knn.predict(X_test_scaled)
    nb_pred  = nb.predict(X_test)

    acc = {
        "Logistic Regression": round(accuracy_score(Y_test, lr_pred)  * 100, 2),
        "KNN":                  round(accuracy_score(Y_test, knn_pred) * 100, 2),
        "Naive Bayes":          round(accuracy_score(Y_test, nb_pred)  * 100, 2),
    }
    prec = {
        "Logistic Regression": round(precision_score(Y_test, lr_pred,  average="weighted") * 100, 2),
        "KNN":                  round(precision_score(Y_test, knn_pred, average="weighted") * 100, 2),
        "Naive Bayes":          round(precision_score(Y_test, nb_pred,  average="weighted") * 100, 2),
    }
    cms = {
        "Logistic Regression": confusion_matrix(Y_test, lr_pred),
        "KNN":                  confusion_matrix(Y_test, knn_pred),
        "Naive Bayes":          confusion_matrix(Y_test, nb_pred),
    }
    raw_df = pd.read_csv(os.path.join(BASE_DIR, "Iris.csv"))
    return lr, knn, nb, scaler, le, acc, prec, cms, X_train, X_test, Y_train, Y_test, X_train_scaled, X_test_scaled, raw_df

lr_model, knn_model, nb_model, scaler, le, acc, prec, cms, X_train, X_test, Y_train, Y_test, X_train_scaled, X_test_scaled, raw_df = load_models()

SPECIES_INFO = {
    "Iris-setosa":     {"emoji": "🌼", "color": "#E8C4B8", "desc": "Compact & distinct. Small petals, easily separable from other species."},
    "Iris-versicolor": {"emoji": "🌺", "color": "#C4B8D4", "desc": "The 'Blue Flag' iris. Moderate dimensions, found in North America."},
    "Iris-virginica":  {"emoji": "🌷", "color": "#B8D4C4", "desc": "The largest of the three. Long petals, found in Eastern US."},
}

FOREST  = "#1C3A2E"
GOLD    = "#B8953A"
CREAM   = "#F5F0E8"
SAGE    = "#4A7C59"
PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="DM Sans", color="#1A1A1A"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
  <div class="hero-eyebrow">🌿 Botanical Intelligence · Supervised ML</div>
  <h1 class="hero-title">Iris Species<br><em>Classifier</em></h1>
  <p class="hero-sub">
    A machine learning system trained on 150 botanical specimens to identify
    <em>Iris</em> species from four physical measurements with exceptional accuracy.
  </p>
  <div class="hero-badges">
    <span class="badge">🧬 3 ML Models</span>
    <span class="badge">📊 150 Samples</span>
    <span class="badge">🎯 Up to 100% Accuracy</span>
    <span class="badge">Python · scikit-learn</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── SPECIES OVERVIEW CARDS ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
for col, (name, info) in zip([col1, col2, col3], SPECIES_INFO.items()):
    with col:
        st.markdown(f"""
        <div class="species-card" style="border-top: 3px solid {info['color']};">
          <div class="species-icon">{info['emoji']}</div>
          <div class="species-name">{name.replace('Iris-', 'Iris ')}</div>
          <div class="species-desc">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN TABS ────────────────────────────────────────────────────────────────
tab_predict, tab_explore, tab_models = st.tabs(
    ["  🔬  Predict Species  ", "  📊  Explore Data  ", "  🏆  Model Analysis  "]
)

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Configure Prediction</div>', unsafe_allow_html=True)
        st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

        model_choice = st.selectbox(
            "Classification Algorithm",
            ["Logistic Regression", "KNN", "Naive Bayes"],
            help="All three models are trained and ready. Logistic Regression achieved 100% test accuracy."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Sepal Measurements**")
        s_len = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        s_wid = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
        st.markdown("<br>**Petal Measurements**", unsafe_allow_html=True)
        p_len = st.slider("Petal Length (cm)", 1.0, 7.0, 4.4, 0.1)
        p_wid = st.slider("Petal Width (cm)",  0.1, 2.5, 1.4, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("✦ Identify Species", use_container_width=True, type="primary")

    with right:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

        if predict_btn:
            raw_input = np.array([[s_len, s_wid, p_len, p_wid]])
            input_df  = pd.DataFrame(raw_input, columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])

            if model_choice == "Logistic Regression":
                scaled = scaler.transform(input_df)
                pred   = lr_model.predict(scaled)[0]
                proba  = lr_model.predict_proba(scaled)[0]
            elif model_choice == "KNN":
                scaled = scaler.transform(input_df)
                pred   = knn_model.predict(scaled)[0]
                proba  = knn_model.predict_proba(scaled)[0]
            else:
                pred   = nb_model.predict(input_df)[0]
                proba  = nb_model.predict_proba(input_df)[0]

            species = le.inverse_transform([pred])[0]
            info    = SPECIES_INFO[species]
            conf    = round(proba[pred] * 100, 1)

            st.markdown(f"""
            <div class="result-box">
              <span class="result-emoji">{info['emoji']}</span>
              <div class="result-label">Identified Species</div>
              <div class="result-species">{species.replace('Iris-', 'Iris ')}</div>
              <div class="result-confidence">Confidence Score</div>
              <span class="confidence-pill">{conf}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Probability donut chart
            fig = go.Figure(go.Bar(
                x=[round(p*100, 1) for p in proba],
                y=le.classes_,
                orientation='h',
                marker=dict(
                    color=[GOLD if cls == species else "#C8D8C0" for cls in le.classes_],
                    line=dict(color='white', width=2)
                ),
                text=[f"{round(p*100,1)}%" for p in proba],
                textposition='outside',
                textfont=dict(size=12, color=FOREST)
            ))
            fig.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title=dict(text="Class Probability Distribution", font=dict(size=13, color=FOREST)),
                xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=12)),
                height=200,
                margin=dict(l=10, r=60, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Input summary
            st.markdown("**Your Input Summary**")
            summary_cols = st.columns(4)
            labels = ["Sepal L.", "Sepal W.", "Petal L.", "Petal W."]
            vals   = [s_len, s_wid, p_len, p_wid]
            for c, l, v in zip(summary_cols, labels, vals):
                with c:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value" style="font-size:1.3rem">{v}</div>
                      <div class="metric-label">{l}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: white;
                border: 1.5px dashed rgba(184,149,58,0.35);
                border-radius: 16px;
                padding: 3rem 2rem;
                text-align: center;
                color: #5A5A4A;
            ">
              <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
              <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; color: #1C3A2E; margin-bottom: 0.5rem;">
                Awaiting Measurements
              </div>
              <div style="font-size: 0.85rem; line-height: 1.6;">
                Adjust the sliders on the left to set your flower's dimensions,
                then click <strong>Identify Species</strong> to get a prediction.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE DATA
# ════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown('<div class="section-title">Dataset Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">150 samples · 3 species · 4 morphological features</div>', unsafe_allow_html=True)
    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # Quick stats row
    m1, m2, m3, m4 = st.columns(4)
    stats_data = [
        ("150", "Total Samples"),
        ("3", "Species"),
        ("4", "Features"),
        ("50", "Samples / Class"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], stats_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Scatter plot
    display_df = raw_df.copy()
    display_df["Species_clean"] = display_df["Species"].str.replace("Iris-", "")

    scatter_col1, scatter_col2 = st.columns(2)

    with scatter_col1:
        fig_scatter = px.scatter(
            display_df,
            x="PetalLengthCm", y="PetalWidthCm",
            color="Species",
            color_discrete_map={
                "Iris-setosa":     "#E8A090",
                "Iris-versicolor": "#9B8EC4",
                "Iris-virginica":  "#5A9E7A",
            },
            title="Petal Length vs Width",
            labels={"PetalLengthCm": "Petal Length (cm)", "PetalWidthCm": "Petal Width (cm)"},
            template="simple_white",
            hover_data={"Species": False, "Species_clean": True}
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color='white')))
        fig_scatter.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
                                   legend=dict(title="", orientation="h", y=-0.2))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with scatter_col2:
        fig_scatter2 = px.scatter(
            display_df,
            x="SepalLengthCm", y="SepalWidthCm",
            color="Species",
            color_discrete_map={
                "Iris-setosa":     "#E8A090",
                "Iris-versicolor": "#9B8EC4",
                "Iris-virginica":  "#5A9E7A",
            },
            title="Sepal Length vs Width",
            labels={"SepalLengthCm": "Sepal Length (cm)", "SepalWidthCm": "Sepal Width (cm)"},
            template="simple_white",
        )
        fig_scatter2.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color='white')))
        fig_scatter2.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
                                    legend=dict(title="", orientation="h", y=-0.2))
        st.plotly_chart(fig_scatter2, use_container_width=True)

    # Box plots
    st.markdown("<br>", unsafe_allow_html=True)
    feature_choice = st.selectbox(
        "Feature Distribution",
        ["PetalLengthCm", "PetalWidthCm", "SepalLengthCm", "SepalWidthCm"],
        format_func=lambda x: x.replace("Cm", " (cm)").replace("Sepal", "Sepal ").replace("Petal", "Petal ")
    )
    fig_box = px.box(
        display_df,
        x="Species", y=feature_choice,
        color="Species",
        color_discrete_map={
            "Iris-setosa":     "#E8A090",
            "Iris-versicolor": "#9B8EC4",
            "Iris-virginica":  "#5A9E7A",
        },
        title=f"Distribution of {feature_choice.replace('Cm', ' (cm)')}",
        template="simple_white",
        points="all"
    )
    fig_box.update_layout(**PLOTLY_TEMPLATE['layout'], height=340, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Raw data table
    with st.expander("📋 View Raw Dataset"):
        disp = raw_df.copy()
        disp["Species"] = disp["Species"].str.replace("Iris-", "Iris ")
        st.dataframe(disp, use_container_width=True, height=300)
        st.caption(f"150 rows × {len(raw_df.columns)} columns")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tab_models:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comparison across 3 classifiers · 70% test split · 105 test samples</div>', unsafe_allow_html=True)
    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    # Accuracy metric cards
    m1, m2, m3 = st.columns(3)
    model_colors = {"Logistic Regression": FOREST, "KNN": SAGE, "Naive Bayes": "#7A6B4A"}
    for col, (model_name, color) in zip([m1, m2, m3], model_colors.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
              <div style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#5A5A4A; margin-bottom:0.3rem;">{model_name}</div>
              <div class="metric-value" style="color:{color};">{acc[model_name]}%</div>
              <div class="metric-label">Accuracy</div>
              <div style="margin-top:0.5rem; font-size:0.8rem; color:#5A5A4A;">Precision: {prec[model_name]}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar comparison chart
    fig_compare = go.Figure()
    model_names = list(acc.keys())
    acc_vals  = [acc[m] for m in model_names]
    prec_vals = [prec[m] for m in model_names]

    fig_compare.add_trace(go.Bar(
        name="Accuracy",
        x=model_names, y=acc_vals,
        marker_color=[FOREST, SAGE, "#9B8B5A"],
        text=[f"{v}%" for v in acc_vals],
        textposition="outside",
        width=0.35,
        offsetgroup=0
    ))
    fig_compare.add_trace(go.Bar(
        name="Precision",
        x=model_names, y=prec_vals,
        marker_color=["rgba(28,58,46,0.4)", "rgba(74,124,89,0.4)", "rgba(155,139,90,0.4)"],
        text=[f"{v}%" for v in prec_vals],
        textposition="outside",
        width=0.35,
        offsetgroup=1
    ))
    fig_compare.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        barmode='group',
        title="Accuracy & Precision by Model",
        yaxis=dict(range=[94, 103], showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        legend=dict(orientation="h", y=1.12),
        height=340,
        bargap=0.25,
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Confusion Matrix
    st.markdown("<br>")
    st.markdown("**Confusion Matrix**")
    cm_model = st.selectbox("Select model for confusion matrix", list(cms.keys()), key="cm_sel")

    cm = cms[cm_model]
    species_labels = [s.replace("Iris-","") for s in le.classes_]
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=species_labels,
        y=species_labels,
        colorscale=[[0, CREAM], [0.5, "#7DAA8A"], [1, FOREST]],
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        showscale=False,
    ))
    fig_cm.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=f"Confusion Matrix — {cm_model}",
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        height=340,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Model descriptions
    with st.expander("📖 About the Algorithms"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Logistic Regression**")
            st.markdown("A linear classifier that models class probabilities using the softmax function. Works best when features are linearly separable — which they largely are in the Iris dataset. Requires feature scaling.")
        with b:
            st.markdown("**K-Nearest Neighbours**")
            st.markdown("Classifies each sample by majority vote among its k=5 nearest neighbours in feature space. Intuitive and non-parametric, but sensitive to scale — hence StandardScaler is applied first.")
        with c:
            st.markdown("**Naive Bayes**")
            st.markdown("A probabilistic classifier assuming feature independence (Gaussian NB). Surprisingly effective despite its simplicity. No scaling required since it models feature distributions directly.")


# ── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  🌿 &nbsp; Iris Species Classifier &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; 150 samples, 3 species &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
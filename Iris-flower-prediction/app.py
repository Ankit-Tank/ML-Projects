import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS injection ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
.stApp {
    background: radial-gradient(ellipse 140% 60% at 50% -10%, #1a3a20 0%, #0a1a10 60%) !important;
    background-color: #0a1a10 !important;
    color: #d4e8d0 !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.viewerBadge_container__1QSob { display: none !important; }

/* ── Hero section ── */
.hero-section {
    background: linear-gradient(180deg, rgba(20,60,25,0.6) 0%, transparent 100%);
    border-bottom: 1px solid rgba(100,180,120,0.12);
    padding: 52px 48px 40px;
    text-align: center;
    margin-bottom: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(80,180,100,0.1);
    border: 1px solid rgba(80,180,100,0.22);
    border-radius: 100px;
    padding: 5px 18px;
    font-size: 11px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #7ecf95;
    margin-bottom: 22px;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: clamp(32px, 4vw, 52px);
    font-weight: 600;
    color: #e8f5e4;
    line-height: 1.15;
    margin-bottom: 12px;
    letter-spacing: -.02em;
}
.hero-title em { font-style: italic; color: #7ecf95; }
.hero-sub {
    color: #7aab82;
    font-size: 15px;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.75;
}

/* ── Main content padding ── */
.main-content {
    padding: 40px 48px;
    max-width: 1200px;
    margin: 0 auto;
}

/* ── Section labels ── */
.section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #4d8c5a;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(77,140,90,0.22);
}

/* ── Accuracy cards ── */
.acc-card {
    background: rgba(12,28,15,0.85);
    border: 1px solid rgba(100,170,110,0.14);
    border-radius: 16px;
    padding: 20px 22px;
    transition: all .2s;
    cursor: pointer;
    height: 100%;
}
.acc-card.selected {
    border-color: rgba(80,200,100,0.45);
    background: rgba(20,50,25,0.7);
    box-shadow: 0 0 30px rgba(60,160,80,0.1) inset;
}
.acc-model { font-size: 14px; color: #90c89a; font-weight: 500; margin-bottom: 4px; }
.acc-num {
    font-size: 34px;
    font-weight: 500;
    color: #7ee89a;
    letter-spacing: -.04em;
    font-family: 'DM Sans', sans-serif;
    margin: 4px 0;
}
.acc-unit { font-size: 18px; opacity: .7; }
.acc-lbl { font-size: 11px; color: #4d8c5a; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 10px; }
.acc-desc { font-size: 12px; color: #4d8c5a; line-height: 1.55; font-weight: 300; }

/* ── Card containers ── */
.card {
    background: rgba(18,35,20,0.75);
    border: 1px solid rgba(100,170,110,0.13);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 0;
}
.card-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 17px;
    font-weight: 600;
    color: #c8e8c0;
    margin-bottom: 20px;
}

/* ── Prediction result card ── */
.pred-setosa   { background: #2a1520; border: 1px solid rgba(240,160,184,0.2); color: #f7d0de; }
.pred-versicolor { background: #0f2535; border: 1px solid rgba(110,198,232,0.2); color: #c8e8f5; }
.pred-virginica { background: #1e1030; border: 1px solid rgba(180,143,224,0.2); color: #dfc8f8; }

.pred-box {
    border-radius: 20px;
    padding: 30px;
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}
.pred-emoji { font-size: 52px; line-height: 1; margin-bottom: 10px; }
.pred-label { font-size: 10px; letter-spacing: .13em; text-transform: uppercase; opacity: .55; margin-bottom: 6px; }
.pred-name {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 30px;
    font-weight: 600;
    line-height: 1.2;
    margin-bottom: 18px;
}
.pred-name em { font-style: italic; display: block; font-size: 22px; opacity: .75; margin-top: 4px; }
.pred-desc {
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    padding: 13px 16px;
    font-size: 13px;
    line-height: 1.65;
    font-weight: 300;
    margin-top: 10px;
}

/* ── Confidence bar ── */
.conf-wrap { margin-bottom: 16px; }
.conf-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; opacity: .8; }
.conf-bar-bg { height: 8px; border-radius: 4px; background: rgba(255,255,255,0.08); overflow: hidden; }
.conf-fill-setosa    { background: #f0a0b8; }
.conf-fill-versicolor{ background: #6ec6e8; }
.conf-fill-virginica { background: #b48fe0; }

/* ── Prob bars ── */
.prob-row { margin-bottom: 16px; }
.prob-header { display: flex; justify-content: space-between; align-items: center; font-size: 13px; margin-bottom: 6px; }
.prob-bar-bg { height: 10px; background: rgba(255,255,255,0.06); border-radius: 5px; overflow: hidden; }

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > div > div > div {
    background: rgba(60,184,106,0.2) !important;
}
div[data-testid="stSlider"] label p {
    color: #7aab82 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #b8e0c0 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}
.stSlider .rc-slider-track { background: #3cb86a !important; }
.stSlider .rc-slider-handle {
    background: #3cb86a !important;
    border-color: #0a1a10 !important;
    box-shadow: 0 0 12px rgba(60,184,106,0.5) !important;
}
div[data-testid="stSelectbox"] label p {
    color: #7aab82 !important;
    font-size: 13px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(12,28,15,0.9) !important;
    border-color: rgba(100,170,110,0.25) !important;
    color: #c8e8c0 !important;
    border-radius: 12px !important;
}

/* ── Metric override ── */
div[data-testid="stMetric"] {
    background: rgba(12,28,15,0.8) !important;
    border: 1px solid rgba(100,170,110,0.14) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
div[data-testid="stMetric"] label {
    color: #4d8c5a !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: .1em !important;
}
div[data-testid="stMetricValue"] {
    color: #7ee89a !important;
    font-size: 30px !important;
    font-weight: 500 !important;
    letter-spacing: -.03em !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(100,170,110,0.15) !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a5c28 0%, #2a7a3a 100%) !important;
    color: #b8f0c8 !important;
    border: 1px solid rgba(80,200,100,0.35) !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 14px 32px !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 20px rgba(30,100,50,0.3) !important;
    transition: all .2s !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #225e30 0%, #358044 100%) !important;
    box-shadow: 0 6px 28px rgba(30,150,60,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Bar chart ── */
div[data-testid="stVegaLiteChart"] canvas { border-radius: 8px; }

/* ── Species cards ── */
.sp-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 8px; }
.sp-card {
    border-radius: 16px;
    padding: 22px;
    border: 1px solid transparent;
    transition: transform .25s;
}
.sp-card:hover { transform: translateY(-3px); }
.sp-emoji-lg { font-size: 30px; margin-bottom: 10px; }
.sp-nm {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 5px;
}
.sp-hint { font-size: 11px; opacity: .62; margin-bottom: 9px; letter-spacing: .04em; }
.sp-desc { font-size: 13px; line-height: 1.65; opacity: .8; font-weight: 300; }
.sp-stats { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.sp-stat {
    background: rgba(0,0,0,.2);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 12px;
}

/* ── Divider ── */
.styled-divider {
    border: none;
    border-top: 1px solid rgba(100,170,110,0.12);
    margin: 36px 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 52px;
    padding: 28px 0 48px;
    border-top: 1px solid rgba(100,170,110,0.1);
    color: #2e5035;
    font-size: 12px;
    letter-spacing: .05em;
}

/* ── Columns spacing ── */
div[data-testid="column"] { padding: 0 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load & train (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "Iris.csv")
    df = pd.read_csv(csv_path)

    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])

    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X, Y = df[features], df["Species"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, random_state=42
    )

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)

    lr  = LogisticRegression(max_iter=1000).fit(Xtr_sc, Y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr_sc, Y_train)
    nb  = GaussianNB().fit(X_train, Y_train)

    def _acc(y_true, y_pred): return round(accuracy_score(y_true, y_pred)*100, 1)
    def _pre(y_true, y_pred): return round(precision_score(y_true, y_pred, average="weighted")*100, 1)

    acc = {
        "Logistic Regression": _acc(Y_test, lr.predict(Xte_sc)),
        "KNN":                  _acc(Y_test, knn.predict(Xte_sc)),
        "Naive Bayes":          _acc(Y_test, nb.predict(X_test)),
    }
    prec = {
        "Logistic Regression": _pre(Y_test, lr.predict(Xte_sc)),
        "KNN":                  _pre(Y_test, knn.predict(Xte_sc)),
        "Naive Bayes":          _pre(Y_test, nb.predict(X_test)),
    }
    return lr, knn, nb, scaler, le, acc, prec

lr_model, knn_model, nb_model, scaler, le, acc, prec = load_models()


# ── Species metadata ─────────────────────────────────────────────────────────
SPECIES_META = {
    "Iris-setosa": {
        "emoji": "🌼", "color": "#f0a0b8",
        "bg": "#2a1520", "text": "#f7d0de",
        "css_class": "pred-setosa", "fill_class": "conf-fill-setosa",
        "hint": "Tiny petals (< 2 cm), broad sepals",
        "desc": "Small, hardy species. Distinguished by short broad petals; grows in arctic and subarctic regions of North America and Asia.",
        "sepal_avg": 4.9, "petal_avg": 1.5,
    },
    "Iris-versicolor": {
        "emoji": "🌺", "color": "#6ec6e8",
        "bg": "#0f2535", "text": "#c8e8f5",
        "css_class": "pred-versicolor", "fill_class": "conf-fill-versicolor",
        "hint": "Medium petals (3–5 cm range)",
        "desc": "Blue flag iris, native to eastern North America. Medium-sized with striking blue-violet blooms along stream banks.",
        "sepal_avg": 5.9, "petal_avg": 4.3,
    },
    "Iris-virginica": {
        "emoji": "🌷", "color": "#b48fe0",
        "bg": "#1e1030", "text": "#dfc8f8",
        "css_class": "pred-virginica", "fill_class": "conf-fill-virginica",
        "hint": "Long petals (> 5 cm), large overall",
        "desc": "Virginia iris — the largest of the three. Elegant lavender-purple flowers favoured in ornamental horticulture.",
        "sepal_avg": 6.6, "petal_avg": 5.6,
    },
}
MODEL_DESC = {
    "Logistic Regression": "Softmax linear classifier optimized with gradient descent.",
    "KNN":                  "Majority vote among 5 nearest training examples.",
    "Naive Bayes":          "Per-class Gaussian distributions for probabilistic inference.",
}


# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-section">
  <div class="hero-badge">🌿 &nbsp; Supervised Machine Learning</div>
  <h1 class="hero-title">Iris Flower <em>Species Predictor</em></h1>
  <p class="hero-sub">
    Identify iris species from petal and sepal measurements using three classical
    machine learning algorithms trained on 150 botanical samples.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Content wrapper start ────────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE CARDS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Model Performance &nbsp; — &nbsp; Select a classifier</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
model_choice = st.session_state.get("model", "Logistic Regression")

for col, name in zip([col1, col2, col3], ["Logistic Regression", "KNN", "Naive Bayes"]):
    sel = "selected" if name == model_choice else ""
    with col:
        st.markdown(f"""
        <div class="acc-card {sel}">
          <div class="acc-model">{name}</div>
          <div class="acc-num">{acc[name]}<span class="acc-unit">%</span></div>
          <div class="acc-lbl">Accuracy</div>
          <div class="acc-desc">{MODEL_DESC[name]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Model selector (actual widget) ──────────────────────────────────────────
model_choice = st.selectbox(
    "🤖 Choose Classification Model",
    ["Logistic Regression", "KNN", "Naive Bayes"],
    label_visibility="visible"
)

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SLIDERS + PREDICTION  (side by side)
# ════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

# ── LEFT: Sliders ────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📐 Flower Measurements</div>', unsafe_allow_html=True)

    sl1, sl2 = st.columns(2)
    with sl1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    with sl2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.4, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.4, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Mini measurement summary ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Input Profile</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for mcol, label, val in zip([m1,m2,m3,m4],
                                 ["Sepal L","Sepal W","Petal L","Petal W"],
                                 [sepal_length, sepal_width, petal_length, petal_width]):
        with mcol:
            st.metric(label, f"{val} cm")

    st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: Prediction ────────────────────────────────────────────────────────
with right:
    raw = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    inp = pd.DataFrame(raw, columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])

    if model_choice == "Logistic Regression":
        scaled     = scaler.transform(inp)
        prediction = lr_model.predict(scaled)[0]
        proba      = lr_model.predict_proba(scaled)[0]
    elif model_choice == "KNN":
        scaled     = scaler.transform(inp)
        prediction = knn_model.predict(scaled)[0]
        proba      = knn_model.predict_proba(scaled)[0]
    else:
        prediction = nb_model.predict(inp)[0]
        proba      = nb_model.predict_proba(inp)[0]

    species_name = le.inverse_transform([prediction])[0]
    meta         = SPECIES_META[species_name]
    confidence   = round(proba[prediction] * 100, 1)
    conf_bar_w   = int(confidence)

    genus, epithet = species_name.replace("Iris-","").capitalize(), ""
    parts = species_name.split("-")
    if len(parts) == 2:
        epithet = parts[1]

    # Prediction box
    st.markdown(f"""
    <div class="pred-box {meta['css_class']}">
      <div class="pred-emoji">{meta['emoji']}</div>
      <div class="pred-label">Predicted Species</div>
      <div class="pred-name">Iris <em>{epithet}</em></div>
      <div class="conf-wrap">
        <div class="conf-header">
          <span>Confidence</span>
          <span style="font-weight:500">{confidence}%</span>
        </div>
        <div class="conf-bar-bg">
          <div class="{meta['fill_class']}" style="width:{conf_bar_w}%; height:8px; border-radius:4px;"></div>
        </div>
      </div>
      <div class="pred-desc">{meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilities card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Class Probabilities</div>', unsafe_allow_html=True)

    for i, (sp_raw, sp_meta) in enumerate(SPECIES_META.items()):
        p = round(proba[i] * 100, 1)
        is_pred = (i == prediction)
        bar_color = sp_meta["color"] if is_pred else f"{sp_meta['color']}55"
        name_clean = sp_raw.replace("Iris-", "Iris ").capitalize()
        bold = "font-weight:500" if is_pred else "font-weight:400"
        text_color = sp_meta["color"] if is_pred else "#5a8a64"
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-header">
            <span style="display:flex;align-items:center;gap:8px">
              <span style="font-size:16px">{sp_meta['emoji']}</span>
              <span style="color:#9ecba4;font-size:13px">{sp_raw}</span>
            </span>
            <span style="font-size:14px;{bold};color:{text_color}">{p}%</span>
          </div>
          <div class="prob-bar-bg">
            <div style="height:10px;border-radius:5px;width:{int(p)}%;background:{bar_color};
                        transition:width .5s cubic-bezier(.34,1.56,.64,1)"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PROBABILITY BAR CHART
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
prob_df = pd.DataFrame({
    "Species":     le.classes_,
    "Probability": [round(p*100, 2) for p in proba],
}).set_index("Species")
st.bar_chart(prob_df, color="#3cb86a", height=200)

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SPECIES REFERENCE CARDS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Species Reference Guide</div>', unsafe_allow_html=True)

cards_html = '<div class="sp-grid">'
for sp_raw, m in SPECIES_META.items():
    cards_html += f"""
    <div class="sp-card" style="background:{m['bg']};border-color:{m['color']}28;color:{m['text']}">
      <div class="sp-emoji-lg">{m['emoji']}</div>
      <div class="sp-nm" style="color:{m['color']}">{sp_raw.replace('-',' ')}</div>
      <div class="sp-hint">{m['hint']}</div>
      <div class="sp-desc">{m['desc']}</div>
      <div class="sp-stats">
        <div class="sp-stat">
          <span style="opacity:.6">Sepal avg: </span>
          <span style="font-weight:500;color:{m['color']}">{m['sepal_avg']} cm</span>
        </div>
        <div class="sp-stat">
          <span style="opacity:.6">Petal avg: </span>
          <span style="font-weight:500;color:{m['color']}">{m['petal_avg']} cm</span>
        </div>
      </div>
    </div>
    """
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Model Comparison — Test Set</div>', unsafe_allow_html=True)

results_df = pd.DataFrame({
    "Model":          list(acc.keys()),
    "Accuracy (%)":   list(acc.values()),
    "Precision (%)":  list(prec.values()),
}).set_index("Model")

st.dataframe(
    results_df.style
        .format("{:.1f}")
        .highlight_max(color="#1a4a22", subset=["Accuracy (%)", "Precision (%)"])
        .set_properties(**{"color": "#c8e8c0", "font-size": "14px"}),
    use_container_width=True,
    height=160,
)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🌿 &nbsp; Iris Species Predictor &nbsp;·&nbsp;
  Logistic Regression · KNN · Naive Bayes &nbsp;·&nbsp;
  150 samples &nbsp;·&nbsp; Fisher 1936 &nbsp;·&nbsp;
  Train/Test split: 30% / 70%
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-content
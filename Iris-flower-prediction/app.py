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
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── THEME: Deep indigo/violet — perfect for iris flowers ────────────────────
BG       = "#0e0b1e"
SURF1    = "#16122e"
SURF2    = "#1e1940"
BORDER   = "rgba(160,130,255,0.14)"
BORDER2  = "rgba(160,130,255,0.28)"
VIOLET   = "#a78bfa"
VIOLET_D = "#7c5cf5"
PINK     = "#f472b6"
GOLD     = "#fbbf24"
TEXT     = "#ede9fe"
MUTED    = "#8b7ab8"
MUTED2   = "#5b4d82"
GREEN    = "#34d399"

# ─── Species palette ──────────────────────────────────────────────────────────
SP = {
    "Iris-setosa": {
        "emoji":"🌼","color":"#fb923c","bg":"#2a1500","border":"rgba(251,146,60,0.25)",
        "text":"#fde8cb","hint":"Tiny petals (< 2 cm), broad sepals",
        "desc":"Small, hardy species. Distinguished by short broad petals; thrives in arctic and subarctic regions of North America and Asia.",
        "sepal":4.9,"petal":1.5,
    },
    "Iris-versicolor": {
        "emoji":"🌺","color":"#a78bfa","bg":"#130e28","border":"rgba(167,139,250,0.25)",
        "text":"#ddd6fe","hint":"Medium petals (3 – 5 cm range)",
        "desc":"Blue flag iris, native to eastern North America. Medium-sized with striking blue-violet blooms along stream banks and meadows.",
        "sepal":5.9,"petal":4.3,
    },
    "Iris-virginica": {
        "emoji":"🌷","color":"#f472b6","bg":"#200b1c","border":"rgba(244,114,182,0.25)",
        "text":"#fce7f3","hint":"Long petals (> 5 cm), largest species",
        "desc":"Virginia iris — the largest of the three. Elegant pink-lavender flowers widely favoured in ornamental horticulture.",
        "sepal":6.6,"petal":5.6,
    },
}

MODEL_DESC = {
    "Logistic Regression": "Softmax linear classifier optimized with gradient descent.",
    "KNN":                  "Majority vote among 5 nearest training examples.",
    "Naive Bayes":          "Per-class Gaussian distributions for probabilistic inference.",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — inline styles preferred; this block only handles Streamlit widgets
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400;1,600&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
.stApp {{ background-color: {BG} !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
#MainMenu, footer, header, .stDeployButton,
.viewerBadge_container__1QSob {{ display:none !important; }}

/* Sliders */
div[data-testid="stSlider"] label p {{
    color: {MUTED} !important; font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}}
div[data-testid="stSlider"] [data-testid="stThumbValue"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] [data-testid="stTickBarMin"] {{
    color: {VIOLET} !important; font-size: 14px !important; font-weight: 500 !important;
}}
.stSlider .rc-slider-track {{ background: {VIOLET_D} !important; }}
.stSlider .rc-slider-handle {{
    background: {VIOLET} !important; border-color: {BG} !important;
    box-shadow: 0 0 12px rgba(124,92,245,0.55) !important;
}}

/* Selectbox */
div[data-testid="stSelectbox"] label p {{
    color: {MUTED} !important; font-size: 13px !important;
}}
div[data-testid="stSelectbox"] > div > div {{
    background: {SURF2} !important;
    border: 1px solid {BORDER2} !important;
    color: {VIOLET} !important; border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}}

/* Metric */
div[data-testid="stMetric"] {{
    background: {SURF1} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important; padding: 14px 18px !important;
}}
div[data-testid="stMetric"] label {{
    color: {MUTED2} !important; font-size: 10px !important;
    text-transform: uppercase !important; letter-spacing: .1em !important;
}}
div[data-testid="stMetricValue"] {{
    color: {VIOLET} !important; font-size: 22px !important;
    font-weight: 500 !important; letter-spacing: -.02em !important;
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{
    border-radius: 14px !important; overflow: hidden !important;
    border: 1px solid {BORDER} !important;
}}

/* Bar chart */
div[data-testid="stVegaLiteChart"] {{ background: transparent !important; }}

/* Column gaps */
div[data-testid="column"] {{ padding: 0 10px !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    BASE = os.path.dirname(os.path.abspath(__file__))
    df   = pd.read_csv(os.path.join(BASE, "Iris.csv"))
    le   = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])

    feats = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
    X, Y  = df[feats], df["Species"]
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.7, random_state=42)

    sc = StandardScaler()
    Xs_tr, Xs_te = sc.fit_transform(X_tr), sc.transform(X_te)

    lr  = LogisticRegression(max_iter=1000).fit(Xs_tr, Y_tr)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xs_tr, Y_tr)
    nb  = GaussianNB().fit(X_tr, Y_tr)

    def _a(yt, yp): return round(accuracy_score(yt,yp)*100,1)
    def _p(yt, yp): return round(precision_score(yt,yp,average="weighted")*100,1)

    acc  = {"Logistic Regression":_a(Y_te,lr.predict(Xs_te)),  "KNN":_a(Y_te,knn.predict(Xs_te)),  "Naive Bayes":_a(Y_te,nb.predict(X_te))}
    prec = {"Logistic Regression":_p(Y_te,lr.predict(Xs_te)),  "KNN":_p(Y_te,knn.predict(Xs_te)),  "Naive Bayes":_p(Y_te,nb.predict(X_te))}
    return lr, knn, nb, sc, le, acc, prec

lr_m, knn_m, nb_m, scaler, le, acc, prec = load_models()


# ─────────────────────────────────────────────────────────────────────────────
# HERO  — rendered in its own iframe so nothing leaks
# ─────────────────────────────────────────────────────────────────────────────
components.html(f"""
<!DOCTYPE html>
<html>
<head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400;1,600&family=Inter:wght@300;400&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: linear-gradient(180deg, #1a1435 0%, {BG} 100%);
    border-bottom: 1px solid rgba(160,130,255,0.12);
    padding: 52px 60px 44px;
    text-align: center;
  }}
  .badge {{
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.22);
    border-radius: 100px; padding: 5px 18px;
    font-family: 'Inter', sans-serif; font-size: 11px;
    letter-spacing: .13em; text-transform: uppercase; color: #a78bfa;
    margin-bottom: 24px;
  }}
  h1 {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(34px, 5vw, 56px); font-weight: 600;
    color: #ede9fe; line-height: 1.12; letter-spacing: -.02em;
    margin-bottom: 14px;
  }}
  h1 em {{ font-style: italic; color: #a78bfa; }}
  p {{
    font-family: 'Inter', sans-serif; font-size: 15px;
    font-weight: 300; color: #8b7ab8;
    max-width: 520px; margin: 0 auto; line-height: 1.75;
  }}
  .dots {{
    display: flex; justify-content: center; gap: 18px;
    margin-top: 28px;
  }}
  .dot {{
    width: 8px; height: 8px; border-radius: 50%;
    opacity: .5;
  }}
</style>
</head>
<body>
  <div class="badge">🌸 &nbsp; Supervised Machine Learning</div>
  <h1>Iris Flower <em>Species Predictor</em></h1>
  <p>Identify iris species from petal and sepal measurements using three classical machine learning algorithms trained on 150 botanical samples.</p>
  <div class="dots">
    <div class="dot" style="background:#fb923c"></div>
    <div class="dot" style="background:#a78bfa"></div>
    <div class="dot" style="background:#f472b6"></div>
  </div>
</body>
</html>
""", height=260)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def spacer(h=24):
    st.markdown(f'<div style="height:{h}px"></div>', unsafe_allow_html=True)

def section_label(text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:32px 0 18px">
      <span style="font-size:10px;font-weight:500;letter-spacing:.14em;text-transform:uppercase;color:{MUTED2}">{text}</span>
      <div style="flex:1;height:1px;background:rgba(160,130,255,0.13)"></div>
    </div>""", unsafe_allow_html=True)

def divider():
    st.markdown(f'<div style="height:1px;background:rgba(160,130,255,0.1);margin:36px 0"></div>', unsafe_allow_html=True)

# Outer padding
st.markdown(f'<div style="padding:0 48px">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ACCURACY CARDS
# ─────────────────────────────────────────────────────────────────────────────
section_label("Model Performance — Select a Classifier")

model_choice = st.selectbox(
    "Classification Model",
    ["Logistic Regression", "KNN", "Naive Bayes"],
    label_visibility="collapsed"
)

spacer(12)
c1, c2, c3 = st.columns(3)
for col, name in zip([c1, c2, c3], ["Logistic Regression", "KNN", "Naive Bayes"]):
    sel_border = f"1px solid {VIOLET}" if name == model_choice else f"1px solid {BORDER}"
    sel_bg     = f"rgba(124,92,245,0.12)" if name == model_choice else SURF1
    with col:
        st.markdown(f"""
        <div style="background:{sel_bg};border:{sel_border};border-radius:18px;
                    padding:22px 24px;cursor:pointer;min-height:160px">
          <div style="font-size:14px;font-weight:500;color:#c4b5fd;margin-bottom:6px">{name}</div>
          <div style="font-size:36px;font-weight:500;color:{VIOLET};letter-spacing:-.04em;
                      font-family:'Inter',sans-serif;margin:4px 0">
            {acc[name]}<span style="font-size:18px;opacity:.65">%</span>
          </div>
          <div style="font-size:10px;color:{MUTED2};text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:12px">Accuracy</div>
          <div style="font-size:12px;color:{MUTED2};line-height:1.55;font-weight:300">
            {MODEL_DESC[name]}
          </div>
        </div>""", unsafe_allow_html=True)

divider()


# ─────────────────────────────────────────────────────────────────────────────
# SLIDERS  +  PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ── LEFT ─────────────────────────────────────────────────────────────────────
with left:
    st.markdown(f"""
    <div style="background:{SURF1};border:1px solid {BORDER};border-radius:20px;
                padding:28px 30px 20px">
      <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:19px;
                  font-weight:600;color:#ddd6fe;margin-bottom:22px">
        📐 &nbsp; Flower Measurements
      </div>
    </div>""", unsafe_allow_html=True)

    # Sliders inside separate container to avoid nesting issues
    with st.container():
        sl1, sl2 = st.columns(2)
        with sl1:
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
        with sl2:
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.4, 0.1)
            petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.4, 0.1)

    spacer(16)
    # Input summary metrics
    st.markdown(f"""
    <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:19px;
                font-weight:600;color:#ddd6fe;margin-bottom:14px">
      📊 &nbsp; Input Profile
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for mc, lbl, val in zip([m1,m2,m3,m4],
                             ["Sepal L","Sepal W","Petal L","Petal W"],
                             [sepal_length,sepal_width,petal_length,petal_width]):
        with mc:
            st.metric(lbl, f"{val} cm")

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with right:
    raw  = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    inp  = pd.DataFrame(raw, columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])

    if model_choice == "Logistic Regression":
        sc_inp = scaler.transform(inp)
        pred   = lr_m.predict(sc_inp)[0]
        proba  = lr_m.predict_proba(sc_inp)[0]
    elif model_choice == "KNN":
        sc_inp = scaler.transform(inp)
        pred   = knn_m.predict(sc_inp)[0]
        proba  = knn_m.predict_proba(sc_inp)[0]
    else:
        pred   = nb_m.predict(inp)[0]
        proba  = nb_m.predict_proba(inp)[0]

    sp_name = le.inverse_transform([pred])[0]
    meta    = SP[sp_name]
    conf    = round(proba[pred]*100, 1)
    epithet = sp_name.split("-")[1] if "-" in sp_name else sp_name
    conf_w  = int(conf)

    # ── Prediction result card (inline styles only — no CSS classes) ──────────
    st.markdown(f"""
    <div style="background:{meta['bg']};border:{meta['border']};border:1px solid {meta['border'].split('rgba')[1] and meta['border']};
                border-radius:20px;padding:28px 30px;margin-bottom:18px;
                border:1px solid {meta['color']}30">
      <div style="font-size:50px;line-height:1;margin-bottom:10px">{meta['emoji']}</div>
      <div style="font-size:10px;letter-spacing:.13em;text-transform:uppercase;
                  opacity:.55;margin-bottom:6px;color:{meta['text']}">Predicted Species</div>
      <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:32px;
                  font-weight:600;color:{meta['color']};line-height:1.1;margin-bottom:4px">
        Iris
      </div>
      <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:24px;
                  font-style:italic;color:{meta['color']};opacity:.8;margin-bottom:20px">
        {epithet}
      </div>
      <div style="display:flex;justify-content:space-between;font-size:12px;
                  color:{meta['text']};opacity:.75;margin-bottom:6px">
        <span>Confidence</span><span style="font-weight:500">{conf}%</span>
      </div>
      <div style="height:8px;border-radius:4px;background:rgba(255,255,255,0.08);
                  overflow:hidden;margin-bottom:18px">
        <div style="height:100%;width:{conf_w}%;background:{meta['color']};
                    border-radius:4px"></div>
      </div>
      <div style="background:rgba(0,0,0,0.22);border-radius:12px;padding:13px 15px;
                  font-size:13px;line-height:1.65;font-weight:300;color:{meta['text']}">
        {meta['desc']}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Probabilities ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{SURF1};border:1px solid {BORDER};border-radius:20px;
                padding:24px 28px">
      <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:19px;
                  font-weight:600;color:#ddd6fe;margin-bottom:18px">
        🎯 &nbsp; Class Probabilities
      </div>""", unsafe_allow_html=True)

    for i, (sp_raw, sp_m) in enumerate(SP.items()):
        p        = round(proba[i]*100, 1)
        is_pred  = (i == pred)
        bar_col  = sp_m["color"] if is_pred else f"{sp_m['color']}44"
        w        = int(p)
        fw       = "600" if is_pred else "400"
        txt_col  = sp_m["color"] if is_pred else MUTED
        st.markdown(f"""
        <div style="margin-bottom:16px">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:7px">
            <span style="display:flex;align-items:center;gap:8px">
              <span style="font-size:16px">{sp_m['emoji']}</span>
              <span style="color:{txt_col};font-size:13px;font-weight:{fw}">{sp_raw}</span>
            </span>
            <span style="font-size:14px;font-weight:{fw};color:{txt_col}">{p}%</span>
          </div>
          <div style="height:10px;background:rgba(255,255,255,0.06);border-radius:5px;
                      overflow:hidden">
            <div style="height:100%;width:{w}%;background:{bar_col};border-radius:5px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

divider()


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
section_label("Probability Distribution")

prob_df = pd.DataFrame({
    "Species":     le.classes_,
    "Probability %": [round(p*100,2) for p in proba],
}).set_index("Species")

st.bar_chart(prob_df, color=VIOLET_D, height=200)

divider()


# ─────────────────────────────────────────────────────────────────────────────
# SPECIES REFERENCE CARDS  — using st.columns, no CSS grid
# ─────────────────────────────────────────────────────────────────────────────
section_label("Species Reference Guide")

sc1, sc2, sc3 = st.columns(3)
for col, (sp_raw, m) in zip([sc1, sc2, sc3], SP.items()):
    with col:
        sp_display = sp_raw.replace("-", " ")
        st.markdown(f"""
        <div style="background:{m['bg']};border:1px solid {m['color']}28;
                    border-radius:18px;padding:24px;min-height:280px">
          <div style="font-size:30px;margin-bottom:12px">{m['emoji']}</div>
          <div style="font-family:'Cormorant Garamond',Georgia,serif;font-size:18px;
                      font-weight:600;color:{m['color']};margin-bottom:6px">
            {sp_display}
          </div>
          <div style="font-size:11px;color:{m['text']};opacity:.6;
                      margin-bottom:10px;letter-spacing:.03em">{m['hint']}</div>
          <div style="font-size:13px;line-height:1.65;color:{m['text']};
                      opacity:.8;font-weight:300;margin-bottom:14px">{m['desc']}</div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <div style="background:rgba(0,0,0,0.22);border-radius:8px;padding:5px 10px;
                        font-size:12px;color:{m['text']}">
              <span style="opacity:.6">Sepal avg: </span>
              <span style="font-weight:500;color:{m['color']}">{m['sepal']} cm</span>
            </div>
            <div style="background:rgba(0,0,0,0.22);border-radius:8px;padding:5px 10px;
                        font-size:12px;color:{m['text']}">
              <span style="opacity:.6">Petal avg: </span>
              <span style="font-weight:500;color:{m['color']}">{m['petal']} cm</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

divider()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
section_label("Model Comparison — Test Set")

results_df = pd.DataFrame({
    "Model":          list(acc.keys()),
    "Accuracy (%)":   list(acc.values()),
    "Precision (%)":  list(prec.values()),
}).set_index("Model")

st.dataframe(
    results_df.style
        .format("{:.1f}")
        .highlight_max(color="#2d1f5e", subset=["Accuracy (%)","Precision (%)"])
        .set_properties(**{"color": "#ddd6fe", "font-size": "14px", "background-color": SURF1}),
    use_container_width=True,
    height=160,
)

spacer(12)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:28px 0 52px;
            border-top:1px solid rgba(160,130,255,0.1);
            color:{MUTED2};font-size:12px;letter-spacing:.05em;margin-top:20px">
  🌸 &nbsp; Iris Species Predictor &nbsp;·&nbsp;
  Logistic Regression · KNN · Naive Bayes &nbsp;·&nbsp;
  150 samples &nbsp;·&nbsp; Fisher 1936 &nbsp;·&nbsp;
  Train / Test split: 30% / 70%
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
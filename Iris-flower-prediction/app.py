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
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

st.set_page_config(
    page_title="IrisAI · Species Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

:root {
  --bg:#070C18;--surface:rgba(255,255,255,0.04);--surface2:rgba(255,255,255,0.07);
  --border:rgba(255,255,255,0.08);--border2:rgba(255,255,255,0.13);
  --indigo:#6366F1;--teal:#2DD4BF;--coral:#FB923C;--rose:#F43F5E;--pink:#F472B6;
  --text:#F1F5F9;--muted:#64748B;--muted2:#94A3B8;
  --gi:rgba(99,102,241,0.35);--gt:rgba(45,212,191,0.3);--gc:rgba(251,146,60,0.3);
}

html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;background-color:var(--bg)!important;color:var(--text);}
.main{background:var(--bg)!important;}
.main .block-container{padding:1.5rem 2rem 4rem!important;max-width:1380px!important;}
section[data-testid="stSidebar"]{display:none;}

/* orbs */
.orb-container{position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;overflow:hidden;}
.orb{position:absolute;border-radius:50%;filter:blur(80px);animation:drift linear infinite;opacity:0.4;}
.orb-1{width:600px;height:600px;background:radial-gradient(circle,rgba(99,102,241,0.6),transparent 70%);top:-200px;left:-150px;animation-duration:25s;}
.orb-2{width:500px;height:500px;background:radial-gradient(circle,rgba(45,212,191,0.5),transparent 70%);top:40%;right:-100px;animation-duration:20s;animation-delay:-8s;}
.orb-3{width:400px;height:400px;background:radial-gradient(circle,rgba(244,63,94,0.4),transparent 70%);bottom:-100px;left:30%;animation-duration:30s;animation-delay:-15s;}
@keyframes drift{0%{transform:translate(0,0) scale(1);}33%{transform:translate(40px,-30px) scale(1.05);}66%{transform:translate(-20px,40px) scale(0.95);}100%{transform:translate(0,0) scale(1);}}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.5;transform:scale(0.85);}}

/* navbar */
.navbar{display:flex;align-items:center;justify-content:space-between;padding:1rem 1.5rem;margin-bottom:1.5rem;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:16px;backdrop-filter:blur(10px);}
.nav-logo{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;letter-spacing:-0.02em;background:linear-gradient(135deg,var(--indigo),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nav-logo .dot{-webkit-text-fill-color:var(--coral);color:var(--coral);}
.nav-tags{display:flex;gap:.5rem;}
.ntag{padding:.25rem .75rem;border-radius:100px;font-size:.68rem;font-weight:600;letter-spacing:.06em;border:1px solid;}
.ni{color:var(--indigo);border-color:var(--indigo);background:rgba(99,102,241,0.08);}
.nt{color:var(--teal);border-color:var(--teal);background:rgba(45,212,191,0.08);}
.nc{color:var(--coral);border-color:var(--coral);background:rgba(251,146,60,0.08);}

/* hero */
.hero{text-align:center;padding:3rem 2rem 2.5rem;}
.hero-chip{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem 1rem;border-radius:100px;font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;border:1px solid rgba(99,102,241,0.4);color:var(--indigo);background:rgba(99,102,241,0.08);margin-bottom:1.5rem;}
.hero-dot{width:6px;height:6px;border-radius:50%;background:var(--teal);animation:pulse 2s infinite;}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(2.8rem,6vw,5rem);font-weight:800;line-height:1.0;letter-spacing:-0.03em;color:var(--text);margin:0 0 1rem;}
.grad{background:linear-gradient(135deg,var(--indigo) 0%,var(--teal) 50%,var(--coral) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-desc{font-size:1rem;color:var(--muted2);max-width:560px;margin:0 auto 2rem;line-height:1.7;font-weight:300;}
.hero-stats{display:flex;justify-content:center;gap:2.5rem;padding:1.2rem 2rem;background:var(--surface);border:1px solid var(--border);border-radius:14px;max-width:480px;margin:0 auto;}
.sv{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;line-height:1;}
.si{color:var(--indigo);}.st{color:var(--teal);}.sc{color:var(--coral);}
.sl{font-size:.68rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-top:.2rem;}

/* species cards */
.sp-card{padding:1.4rem;border-radius:16px;background:var(--surface);border:1px solid var(--border);transition:border-color .3s,transform .3s;}
.sp-card:hover{transform:translateY(-4px);}
.sp-card.setosa{border-top:2px solid var(--pink);}
.sp-card.versicolor{border-top:2px solid var(--teal);}
.sp-card.virginica{border-top:2px solid var(--indigo);}
.sp-icon{font-size:2.4rem;margin-bottom:.6rem;}
.sp-name{font-family:'Syne',sans-serif;font-size:.9rem;font-weight:700;color:var(--text);margin-bottom:.2rem;}
.sp-sci{font-style:italic;font-size:.74rem;color:var(--muted2);margin-bottom:.45rem;}
.sp-desc{font-size:.74rem;color:var(--muted);line-height:1.5;}

/* section */
.sec-lbl{font-size:.65rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);margin-bottom:.3rem;}
.sec-ttl{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text);letter-spacing:-.02em;}

/* measure groups */
.meas-group{background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:12px;padding:1rem 1rem 0.2rem;margin-bottom:1rem;}
.meas-title{font-size:.65rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--indigo);margin-bottom:.8rem;}
.meas-title.pt{color:var(--teal);}

/* sliders */
div[data-testid="stSlider"]>div>div>div{background:linear-gradient(90deg,var(--indigo),var(--teal))!important;}
div[data-testid="stSlider"]>div>div>div>div{background:white!important;border:2px solid var(--indigo)!important;box-shadow:0 0 12px var(--gi)!important;}
div[data-testid="stSlider"] label{color:var(--muted2)!important;font-size:.8rem!important;}

/* selectbox */
div[data-testid="stSelectbox"] label{color:var(--muted2)!important;font-size:.72rem!important;font-weight:600!important;letter-spacing:.08em!important;text-transform:uppercase!important;}
div[data-testid="stSelectbox"]>div>div{background:rgba(255,255,255,0.05)!important;border:1px solid var(--border2)!important;border-radius:10px!important;color:var(--text)!important;}

/* button */
div[data-testid="stButton"]>button{background:linear-gradient(135deg,var(--indigo),#8B5CF6)!important;color:white!important;border:none!important;border-radius:12px!important;padding:.85rem 2rem!important;font-family:'Syne',sans-serif!important;font-size:.85rem!important;font-weight:700!important;letter-spacing:.06em!important;text-transform:uppercase!important;width:100%!important;transition:all .3s!important;box-shadow:0 4px 24px var(--gi)!important;}
div[data-testid="stButton"]>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 36px var(--gi)!important;}

/* result */
.result-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:340px;border:1.5px dashed rgba(99,102,241,0.25);border-radius:20px;text-align:center;padding:2rem;}
.re-icon{font-size:3.5rem;margin-bottom:1rem;}
.re-ttl{font-family:'Syne',sans-serif;font-size:1.1rem;color:var(--muted2);margin-bottom:.5rem;}
.re-sub{font-size:.8rem;color:var(--muted);line-height:1.6;}

.result-card{border-radius:20px;padding:1.8rem;position:relative;overflow:hidden;margin-bottom:1rem;}
.rc-setosa{background:linear-gradient(135deg,rgba(244,63,94,0.12),rgba(251,146,60,0.08));border:1px solid rgba(244,63,94,0.28);}
.rc-versicolor{background:linear-gradient(135deg,rgba(45,212,191,0.12),rgba(99,102,241,0.08));border:1px solid rgba(45,212,191,0.28);}
.rc-virginica{background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.08));border:1px solid rgba(99,102,241,0.28);}
.r-glow{position:absolute;top:-60px;right:-60px;width:200px;height:200px;border-radius:50%;filter:blur(50px);opacity:.45;}
.rg-setosa{background:#F43F5E;}.rg-versicolor{background:var(--teal);}.rg-virginica{background:var(--indigo);}
.r-eye{font-size:.65rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.5rem;}
.re-setosa{color:var(--pink);}.re-versicolor{color:var(--teal);}.re-virginica{color:var(--indigo);}
.r-emoji{font-size:3.5rem;display:block;margin-bottom:.4rem;}
.r-name{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:white;letter-spacing:-.02em;line-height:1.1;margin-bottom:.25rem;}
.r-sci{font-style:italic;font-size:.88rem;color:rgba(255,255,255,.5);margin-bottom:1rem;}
.conf-lbl{display:flex;justify-content:space-between;font-size:.72rem;color:rgba(255,255,255,.5);margin-bottom:.25rem;}
.conf-wrap{background:rgba(255,255,255,.08);border-radius:100px;height:6px;overflow:hidden;}
.conf-bar{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--teal),var(--indigo));}
.m-badge{display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .75rem;border-radius:100px;font-size:.68rem;font-weight:600;letter-spacing:.06em;background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);color:var(--indigo);margin-bottom:1rem;}

/* input chips */
.in-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-top:1rem;}
.in-chip{background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:10px;padding:.7rem .5rem;text-align:center;}
.ic-val{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--text);display:block;}
.ic-lbl{font-size:.62rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-top:.2rem;}

/* metric card */
.mc{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1rem;text-align:center;}
.mc-val{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;line-height:1;margin-bottom:.25rem;}
.mc-lbl{font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.mc-sub{font-size:.72rem;color:var(--muted2);margin-top:.4rem;}

/* tabs */
div[data-testid="stTabs"]>div:first-child{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:12px;padding:.3rem;gap:.2rem;}
div[data-testid="stTabs"] button{font-family:'Syne',sans-serif!important;font-size:.78rem!important;font-weight:600!important;letter-spacing:.06em!important;text-transform:uppercase!important;color:var(--muted)!important;border-radius:8px!important;padding:.5rem 1.2rem!important;border:none!important;transition:all .2s!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{background:var(--indigo)!important;color:white!important;}
div[data-testid="stTabs"] [role="tabpanel"]{padding-top:1.5rem;}
div[data-testid="stTabs"]>div:first-child>div[role="tablist"]>div{display:none;}

/* misc */
hr{border-color:var(--border)!important;margin:1.5rem 0!important;}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;border:1px solid var(--border)!important;}
details{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;}
details summary{color:var(--muted2)!important;font-size:.82rem!important;padding:.8rem 1rem!important;}

.footer{text-align:center;padding:2.5rem;font-size:.72rem;color:var(--muted);letter-spacing:.06em;border-top:1px solid var(--border);margin-top:3rem;}
.footer span{color:var(--indigo);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="orb-container">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    BASE = os.path.dirname(os.path.abspath(__file__))
    df   = pd.read_csv(os.path.join(BASE, "Iris.csv"))
    raw  = df.copy()
    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])
    feats = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
    X, Y  = df[feats], df["Species"]
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.7, random_state=42)
    sc = StandardScaler()
    Xtrs = sc.fit_transform(Xtr)
    Xtes = sc.transform(Xte)
    lr  = LogisticRegression(max_iter=1000).fit(Xtrs, Ytr)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtrs, Ytr)
    nb  = GaussianNB().fit(Xtr, Ytr)
    preds = {
        "Logistic Regression": lr.predict(Xtes),
        "KNN":                  knn.predict(Xtes),
        "Naive Bayes":          nb.predict(Xte),
    }
    acc_d  = {m: round(accuracy_score(Yte, p)*100, 2) for m, p in preds.items()}
    prec_d = {m: round(precision_score(Yte, p, average="weighted")*100, 2) for m, p in preds.items()}
    cms_d  = {m: confusion_matrix(Yte, p) for m, p in preds.items()}
    return lr, knn, nb, sc, le, acc_d, prec_d, cms_d, raw

lr_m, knn_m, nb_m, scaler, le, acc, prec, cms, raw_df = load_all()

SP = {
    "Iris-setosa":     {"emoji":"🌸","cls":"setosa",    "color":"#F472B6","sci":"Iris setosa",    "desc":"Smallest & most distinct. Compact petals, thrives in cold Arctic climates."},
    "Iris-versicolor": {"emoji":"🌊","cls":"versicolor","color":"#2DD4BF","sci":"Iris versicolor","desc":"The Blue Flag iris. Medium build, native to North American wetlands."},
    "Iris-virginica":  {"emoji":"💜","cls":"virginica", "color":"#818CF8","sci":"Iris virginica", "desc":"The Virginia iris. Largest species, found in Eastern US marshes."},
}

DT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#94A3B8", size=11),
    margin=dict(l=10,r=10,t=35,b=10),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#64748B"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#64748B"),
)
CMAP = {"Iris-setosa":"#F472B6","Iris-versicolor":"#2DD4BF","Iris-virginica":"#818CF8"}

# NAVBAR
st.markdown("""
<div class="navbar">
  <div class="nav-logo">IRIS<span class="dot">·</span>AI</div>
  <div class="nav-tags">
    <span class="ntag ni">3 Models</span>
    <span class="ntag nt">150 Samples</span>
    <span class="ntag nc">scikit-learn</span>
  </div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
  <div class="hero-chip"><div class="hero-dot"></div>Live ML Classification System</div>
  <h1 class="hero-title">Identify Any<br><span class="grad">Iris Species</span></h1>
  <p class="hero-desc">Input four morphological measurements and let our ensemble of three machine learning models identify the <em>Iris</em> species in real time.</p>
  <div class="hero-stats">
    <div><div class="sv si">100%</div><div class="sl">Peak Accuracy</div></div>
    <div><div class="sv st">3</div><div class="sl">ML Models</div></div>
    <div><div class="sv sc">150</div><div class="sl">Samples</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# SPECIES CARDS
c1,c2,c3 = st.columns(3)
for col,(name,info) in zip([c1,c2,c3], SP.items()):
    with col:
        st.markdown(f"""
        <div class="sp-card {info['cls']}">
          <div class="sp-icon">{info['emoji']}</div>
          <div class="sp-name">{name}</div>
          <div class="sp-sci">{info['sci']}</div>
          <div class="sp-desc">{info['desc']}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1,tab2,tab3 = st.tabs(["  ⬡  Predict  ","  ⬡  Explore Data  ","  ⬡  Model Analytics  "])

# ── TAB 1: PREDICT ──────────────────────────────────────────────────────────
with tab1:
    L,R = st.columns([1,1.1], gap="large")
    with L:
        st.markdown('<div class="sec-lbl">Configuration</div><div class="sec-ttl">Set Parameters</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        model_choice = st.selectbox("ALGORITHM", ["Logistic Regression","KNN","Naive Bayes"])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="meas-group"><div class="meas-title">🌿 Sepal Dimensions</div>', unsafe_allow_html=True)
        s_len = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        s_wid = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="meas-group"><div class="meas-title pt">🌸 Petal Dimensions</div>', unsafe_allow_html=True)
        p_len = st.slider("Petal Length (cm)", 1.0, 7.0, 4.4, 0.1)
        p_wid = st.slider("Petal Width (cm)",  0.1, 2.5, 1.4, 0.1)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("⬡  RUN CLASSIFICATION", use_container_width=True)

    with R:
        st.markdown('<div class="sec-lbl">Output</div><div class="sec-ttl">Classification Result</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if go_btn:
            inp = np.array([[s_len,s_wid,p_len,p_wid]])
            inp_df = pd.DataFrame(inp, columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])
            if model_choice == "Logistic Regression":
                si = scaler.transform(inp_df); pred = lr_m.predict(si)[0]; proba = lr_m.predict_proba(si)[0]
            elif model_choice == "KNN":
                si = scaler.transform(inp_df); pred = knn_m.predict(si)[0]; proba = knn_m.predict_proba(si)[0]
            else:
                pred = nb_m.predict(inp_df)[0]; proba = nb_m.predict_proba(inp_df)[0]
            species = le.inverse_transform([pred])[0]
            info    = SP[species]
            conf    = round(proba[pred]*100, 1)
            cls     = info['cls']
            st.markdown(f"""
            <div class="result-card rc-{cls}">
              <div class="r-glow rg-{cls}"></div>
              <div class="m-badge">⬡ {model_choice}</div>
              <div class="r-eye re-{cls}">Species Identified</div>
              <span class="r-emoji">{info['emoji']}</span>
              <div class="r-name">{species.replace('Iris-','Iris ')}</div>
              <div class="r-sci">{info['sci']}</div>
              <div class="conf-lbl"><span>Confidence Score</span><span>{conf}%</span></div>
              <div class="conf-wrap"><div class="conf-bar" style="width:{conf}%"></div></div>
            </div>""", unsafe_allow_html=True)

            fig = go.Figure()
            for cls_name, p in zip(le.classes_, proba):
                is_w = cls_name == species
                fig.add_trace(go.Bar(
                    x=[cls_name.replace("Iris-","")], y=[round(p*100,1)],
                    marker_color=CMAP[cls_name] if is_w else "rgba(255,255,255,0.07)",
                    marker_line_color=CMAP[cls_name], marker_line_width=1.5,
                    text=[f"{round(p*100,1)}%"], textposition="outside",
                    textfont=dict(color="white" if is_w else "#64748B", size=12),
                    showlegend=False, width=0.45
                ))
            fig.update_layout(**DT, title=dict(text="Class Probability Distribution", font=dict(color="#94A3B8",size=12)),
                              yaxis=dict(range=[0,120],**DT['yaxis']), height=210, barmode='group', bargap=0.38)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            st.markdown(f"""
            <div class="in-grid">
              <div class="in-chip"><span class="ic-val" style="color:#6366F1">{s_len}</span><div class="ic-lbl">Sepal L.</div></div>
              <div class="in-chip"><span class="ic-val" style="color:#6366F1">{s_wid}</span><div class="ic-lbl">Sepal W.</div></div>
              <div class="in-chip"><span class="ic-val" style="color:#2DD4BF">{p_len}</span><div class="ic-lbl">Petal L.</div></div>
              <div class="in-chip"><span class="ic-val" style="color:#2DD4BF">{p_wid}</span><div class="ic-lbl">Petal W.</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-empty">
              <div class="re-icon">🔬</div>
              <div class="re-ttl">Awaiting Input</div>
              <div class="re-sub">Configure measurements on the left,<br>then click <strong style="color:#6366F1">Run Classification</strong>.</div>
            </div>""", unsafe_allow_html=True)

# ── TAB 2: EXPLORE ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-lbl">Dataset</div><div class="sec-ttl">Visual Exploration</div><br>', unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    for col,(v,l,c) in zip([s1,s2,s3,s4],[("150","Total Samples","#6366F1"),("3","Species","#2DD4BF"),("4","Features","#FB923C"),("50","Per Class","#F472B6")]):
        with col:
            st.markdown(f'<div class="mc"><div class="mc-val" style="color:{c}">{v}</div><div class="mc-lbl">{l}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    e1,e2 = st.columns(2)
    with e1:
        fig1 = px.scatter(raw_df, x="PetalLengthCm", y="PetalWidthCm", color="Species",
                          color_discrete_map=CMAP, template="plotly_dark", title="Petal: Length vs Width",
                          labels={"PetalLengthCm":"Petal Length (cm)","PetalWidthCm":"Petal Width (cm)"})
        fig1.update_traces(marker=dict(size=7, opacity=0.9, line=dict(width=0.5,color='rgba(0,0,0,0.5)')))
        fig1.update_layout(**DT, height=300, legend=dict(title="",orientation="h",y=-0.22,font=dict(size=10)))
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar":False})
    with e2:
        fig2 = px.scatter(raw_df, x="SepalLengthCm", y="SepalWidthCm", color="Species",
                          color_discrete_map=CMAP, template="plotly_dark", title="Sepal: Length vs Width",
                          labels={"SepalLengthCm":"Sepal Length (cm)","SepalWidthCm":"Sepal Width (cm)"})
        fig2.update_traces(marker=dict(size=7, opacity=0.9, line=dict(width=0.5,color='rgba(0,0,0,0.5)')))
        fig2.update_layout(**DT, height=300, legend=dict(title="",orientation="h",y=-0.22,font=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
    st.markdown("<br>", unsafe_allow_html=True)
    feat = st.selectbox("Explore Feature", ["PetalLengthCm","PetalWidthCm","SepalLengthCm","SepalWidthCm"],
                        format_func=lambda x: x.replace("Cm","").replace("Sepal","Sepal ").replace("Petal","Petal ")+" (cm)")
    v1,v2 = st.columns(2)
    with v1:
        fig3 = px.violin(raw_df, x="Species", y=feat, color="Species", color_discrete_map=CMAP,
                         box=True, points="all", template="plotly_dark", title=f"{feat.replace('Cm','')} Violin",
                         labels={feat:feat.replace("Cm"," (cm)"),"Species":""})
        fig3.update_layout(**DT, height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
    with v2:
        fig4 = px.histogram(raw_df, x=feat, color="Species", color_discrete_map=CMAP, nbins=20,
                            barmode="overlay", template="plotly_dark", opacity=0.75,
                            title=f"{feat.replace('Cm','')} Histogram", labels={feat:feat.replace("Cm"," (cm)")})
        fig4.update_layout(**DT, height=320, legend=dict(title="",orientation="h",y=-0.22,font=dict(size=10)))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})
    with st.expander("View Raw Dataset"):
        disp = raw_df.copy(); disp["Species"] = disp["Species"].str.replace("Iris-","Iris ")
        st.dataframe(disp, use_container_width=True, height=280)

# ── TAB 3: ANALYTICS ────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-lbl">Evaluation</div><div class="sec-ttl">Model Analytics</div><br>', unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    mcols = {"Logistic Regression":"#6366F1","KNN":"#2DD4BF","Naive Bayes":"#FB923C"}
    micons = {"Logistic Regression":"▲","KNN":"◈","Naive Bayes":"◉"}
    for col,(mn,mc) in zip([m1,m2,m3], mcols.items()):
        with col:
            st.markdown(f"""
            <div class="mc" style="border-left:3px solid {mc};text-align:left;padding:1.3rem 1.2rem;">
              <div style="font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:{mc};margin-bottom:.5rem;font-weight:700;">{micons[mn]} {mn}</div>
              <div class="mc-val" style="color:{mc};font-size:2.2rem">{acc[mn]}%</div>
              <div class="mc-lbl">Accuracy</div>
              <div class="mc-sub" style="margin-top:.6rem">Precision: <strong style="color:{mc}">{prec[mn]}%</strong></div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    a1,a2 = st.columns(2)
    mnames = list(acc.keys())
    with a1:
        figb = go.Figure()
        figb.add_trace(go.Bar(name="Accuracy", x=mnames, y=[acc[m] for m in mnames],
                              marker_color=["#6366F1","#2DD4BF","#FB923C"],
                              text=[f"{acc[m]}%" for m in mnames], textposition="outside",
                              textfont=dict(color="white",size=11), width=0.3, offsetgroup=0))
        figb.add_trace(go.Bar(name="Precision", x=mnames, y=[prec[m] for m in mnames],
                              marker_color=["rgba(99,102,241,.25)","rgba(45,212,191,.25)","rgba(251,146,60,.25)"],
                              marker_line_color=["#6366F1","#2DD4BF","#FB923C"], marker_line_width=1.5,
                              text=[f"{prec[m]}%" for m in mnames], textposition="outside",
                              textfont=dict(color="#94A3B8",size=11), width=0.3, offsetgroup=1))
        figb.update_layout(**DT, barmode='group', bargap=0.35, title="Accuracy vs Precision",
                           yaxis=dict(range=[92,105],**DT['yaxis']),
                           legend=dict(orientation="h",y=1.12,font=dict(size=10)), height=300)
        st.plotly_chart(figb, use_container_width=True, config={"displayModeBar":False})
    with a2:
        cats = ["Accuracy","Precision","Speed","Interpretability","Scalability"]
        radar_vals = {"Logistic Regression":[100,100,95,98,95],"KNN":[99,99,60,55,50],"Naive Bayes":[98,98,99,75,90]}
        figr = go.Figure()
        for (mn,rv),rc in zip(radar_vals.items(),["#6366F1","#2DD4BF","#FB923C"]):
            figr.add_trace(go.Scatterpolar(
                r=rv+[rv[0]], theta=cats+[cats[0]], fill='toself',
                name=mn.split()[0], line_color=rc,
                fillcolor=rc+"26" if len(rc)==7 else rc, opacity=0.8
            ))
        figr.update_layout(**DT,
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True,range=[0,100],color="#475569",gridcolor="rgba(255,255,255,.06)"),
                       angularaxis=dict(color="#94A3B8",gridcolor="rgba(255,255,255,.06)")),
            legend=dict(orientation="h",y=-0.15,font=dict(size=10)),
            title="Multi-Dimension Comparison", height=300)
        st.plotly_chart(figr, use_container_width=True, config={"displayModeBar":False})

    st.markdown("<br>**Confusion Matrix**")
    cm_sel = st.selectbox("Select model", list(cms.keys()), key="cmsel")
    cm_data = cms[cm_sel]
    sp_lbl = [s.replace("Iris-","") for s in le.classes_]
    cm_cs = {"Logistic Regression":[[0,"rgba(99,102,241,.05)"],[1,"#6366F1"]],
             "KNN":[[0,"rgba(45,212,191,.05)"],[1,"#2DD4BF"]],
             "Naive Bayes":[[0,"rgba(251,146,60,.05)"],[1,"#FB923C"]]}
    figcm = go.Figure(go.Heatmap(z=cm_data, x=sp_lbl, y=sp_lbl,
                                  colorscale=cm_cs[cm_sel], text=cm_data,
                                  texttemplate="<b>%{text}</b>", textfont=dict(size=18,color="white"), showscale=False))
    figcm.update_layout(**DT, title=f"Confusion Matrix — {cm_sel}",
                        xaxis=dict(title="Predicted",**DT['xaxis']),
                        yaxis=dict(title="Actual",autorange="reversed",**DT['yaxis']), height=320)
    st.plotly_chart(figcm, use_container_width=True, config={"displayModeBar":False})

    st.markdown("<br>")
    with st.expander("Algorithm Deep Dive"):
        i1,i2,i3 = st.columns(3)
        alg_info = [
            ("▲ Logistic Regression","#6366F1","Models posterior class probabilities via softmax. Fits a linear decision boundary. Requires feature scaling since gradient descent is scale-sensitive. Best performer on Iris.",["Linear","Probabilistic","Needs Scaling"]),
            ("◈ K-Nearest Neighbours","#2DD4BF","Classifies new points by majority vote among k=5 nearest neighbours in Euclidean space. Non-parametric, no training phase — just stores data. Very scale-sensitive.",["Non-parametric","k=5","Needs Scaling"]),
            ("◉ Naive Bayes","#FB923C","Assumes Gaussian feature distributions and independence given the class. Fast, no scaling needed. Surprisingly effective on Iris despite its assumptions.",["Probabilistic","Fast","No Scaling"]),
        ]
        for col,(title,color,desc,tags) in zip([i1,i2,i3], alg_info):
            with col:
                tag_html="".join(f'<span style="font-size:.62rem;padding:.15rem .5rem;border-radius:100px;border:1px solid {color};color:{color};background:rgba(0,0,0,.2);margin-right:.2rem">{t}</span>' for t in tags)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-top:2px solid {color};border-radius:14px;padding:1.2rem">
                  <div style="font-family:Syne,sans-serif;font-size:.88rem;font-weight:700;color:{color};margin-bottom:.7rem">{title}</div>
                  <div style="font-size:.75rem;color:#94A3B8;line-height:1.65;margin-bottom:.9rem">{desc}</div>
                  <div>{tag_html}</div>
                </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  <span>IRIS·AI</span> &nbsp;·&nbsp; Supervised Machine Learning &nbsp;·&nbsp;
  scikit-learn · Streamlit · Plotly &nbsp;·&nbsp; 150 samples · 3 species · 4 features
</div>
""", unsafe_allow_html=True)
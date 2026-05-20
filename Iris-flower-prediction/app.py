import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="Iris DeepSight", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

# ── Advanced Custom CSS (Glassmorphism & Neon) ────────────────────────────────
st.markdown("""
<style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0f1d 0%, #151b2b 100%);
    }
    
    /* Hide default header and footer for a cleaner app feel */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sleek typography for titles */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #8b9bb4;
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 210, 255, 0.3);
    }
    
    /* Prediction Text Formatting */
    .pred-emoji { font-size: 5rem; text-shadow: 0 0 20px rgba(255,255,255,0.2); }
    .pred-title { color: #8b9bb4; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0; }
    .pred-name { font-size: 3rem; font-weight: 800; color: #ffffff; margin: 5px 0; }
    .pred-conf { font-size: 1.5rem; color: #00d2ff; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Environment & Data Prep (Cached) ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Iris.csv")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "predict_flower_species_combined.ipynb")

@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])
        return None, None, None, None, df, []

    le = LabelEncoder()
    df["Species_Encoded"] = le.fit_transform(df["Species"])
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    
    X = df[features]
    Y = df["Species_Encoded"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    
    models["Logistic Regression"].fit(X_train_scaled, Y_train)
    models["K-Nearest Neighbors"].fit(X_train_scaled, Y_train)
    models["Naive Bayes"].fit(X_train.values, Y_train) 
    
    accuracies = {
        "Logistic Regression": accuracy_score(Y_test, models["Logistic Regression"].predict(X_test_scaled)),
        "K-Nearest Neighbors": accuracy_score(Y_test, models["K-Nearest Neighbors"].predict(X_test_scaled)),
        "Naive Bayes": accuracy_score(Y_test, models["Naive Bayes"].predict(X_test.values))
    }
    return models, scaler, le, accuracies, df, features

models, scaler, le, accuracies, df, features = load_and_train()

SPECIES_META = {
    "Iris-setosa": {"emoji": "🌸", "color": "#ff7675"},
    "Iris-versicolor": {"emoji": "🌺", "color": "#74b9ff"},
    "Iris-virginica": {"emoji": "🌻", "color": "#55efc4"}
}

if df.empty:
    st.error("⚠️ Dataset not found. Please place 'Iris.csv' in the same folder.")
    st.stop()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔮 DeepSight OS")
    st.markdown("Botanical Neural Interface")
    st.write("---")
    
    # Modern Radio Button Nav
    app_mode = st.radio("System Modules", ["🚀 Live Inference", "📊 Dataset Matrix", "⚙️ Core Architecture"])
    
    st.write("---")
    st.markdown("##### Selected Algorithm")
    model_choice = st.selectbox("", ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"], label_visibility="collapsed")

# ── View 1: Live Inference ────────────────────────────────────────────────────
if app_mode == "🚀 Live Inference":
    st.markdown("<div class='hero-title'>Inference Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Adjust biological parameters to run real-time neural classification.</div>", unsafe_allow_html=True)
    
    col_controls, col_visual = st.columns([1, 1.2], gap="large")
    
    with col_controls:
        st.markdown("#### Morphological Inputs")
        c1, c2 = st.columns(2)
        with c1:
            s_len = st.number_input("Sepal Length", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()), float(df["SepalLengthCm"].mean()), step=0.1)
            p_len = st.number_input("Petal Length", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()), float(df["PetalLengthCm"].mean()), step=0.1)
        with c2:
            s_wid = st.number_input("Sepal Width", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()), float(df["SepalWidthCm"].mean()), step=0.1)
            p_wid = st.number_input("Petal Width", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()), float(df["PetalWidthCm"].mean()), step=0.1)

    with col_visual:
        input_data = pd.DataFrame([[s_len, s_wid, p_len, p_wid]], columns=features)
        
        if model_choice in ["Logistic Regression", "K-Nearest Neighbors"]:
            scaled_data = scaler.transform(input_data.values)
            pred_idx = models[model_choice].predict(scaled_data)[0]
            proba = models[model_choice].predict_proba(scaled_data)[0]
        else:
            pred_idx = models["Naive Bayes"].predict(input_data.values)[0]
            proba = models["Naive Bayes"].predict_proba(input_data.values)[0]
            
        species = le.inverse_transform([pred_idx])[0]
        conf = proba[pred_idx] * 100
        meta = SPECIES_META.get(species, {"emoji": "🌱"})
        clean_name = species.replace('Iris-', '').upper()

        st.markdown(f"""
        <div class="glass-card">
            <div class="pred-title">Identified Species</div>
            <div class="pred-emoji">{meta['emoji']}</div>
            <div class="pred-name">{clean_name}</div>
            <div class="pred-conf">Match: {conf:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.markdown("#### Probability Distribution Matrix")
    prob_df = pd.DataFrame({"Species": [s.replace("Iris-", "").upper() for s in le.classes_], "Probability": proba * 100})
    
    fig = px.bar(prob_df, x="Probability", y="Species", orientation='h', color="Species", 
                 color_discrete_sequence=["#ff7675", "#74b9ff", "#55efc4"], text_auto='.2f')
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="white"), height=250, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 100]), 
        yaxis=dict(showgrid=False), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ── View 2: Dataset Matrix ────────────────────────────────────────────────────
elif app_mode == "📊 Dataset Matrix":
    st.markdown("<div class='hero-title'>Data Synthesis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Exploring the multidimensional space of the botanical dataset.</div>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", df.shape[0])
    m2.metric("Feature Dimensions", len(features))
    m3.metric("Target Vectors", len(le.classes_))
    
    st.write("---")
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown("#### Raw Telemetry")
        st.dataframe(df.drop(columns=["Species_Encoded"]).head(20), height=380, use_container_width=True)
        try:
            with open(CSV_PATH, "rb") as f:
                st.download_button("📥 Extract Raw CSV", f, "Iris.csv", "text/csv", use_container_width=True)
        except: pass
        
    with c2:
        st.markdown("#### 3D Feature Space Mapping")
        fig_3d = px.scatter_3d(df, x='SepalLengthCm', y='PetalLengthCm', z='PetalWidthCm',
                               color='Species', color_discrete_sequence=["#ff7675", "#74b9ff", "#55efc4"], opacity=0.8)
        fig_3d.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_3d, use_container_width=True)

# ── View 3: Core Architecture ─────────────────────────────────────────────────
elif app_mode == "⚙️ Core Architecture":
    st.markdown("<div class='hero-title'>System Architecture</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Backend model benchmarks and implementation logic.</div>", unsafe_allow_html=True)
    
    st.markdown("#### 🏆 Performance Benchmarks")
    cols = st.columns(3)
    cols[0].metric("Logistic Regression", f"{accuracies['Logistic Regression']*100:.2f}%")
    cols[1].metric("K-Nearest Neighbors", f"{accuracies['K-Nearest Neighbors']*100:.2f}%")
    cols[2].metric("Naive Bayes", f"{accuracies['Naive Bayes']*100:.2f}%")
    
    st.write("---")
    
    col_code, col_dl = st.columns([3, 1])
    with col_code:
        st.markdown("#### Engine Implementation")
        st.code("""
# Core processing loop utilizing Scikit-Learn
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# Model Instantiation
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# Training Sequence
for name, model in models.items():
    if name == "Naive Bayes":
        model.fit(X_train.values, Y_train) # Unscaled
    else:
        model.fit(X_train_scaled, Y_train) # Scaled
        """, language="python")
        
    with col_dl:
        st.markdown("#### Source Files")
        st.info("Access the raw Jupyter research file.")
        try:
            with open(NOTEBOOK_PATH, "rb") as nb_file:
                st.download_button("📓 Download .ipynb", nb_file, "predict_flower_species.ipynb", use_container_width=True)
        except:
            st.error("Notebook missing.")
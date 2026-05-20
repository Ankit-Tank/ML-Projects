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

# ── Page Configuration & CSS ──────────────────────────────────────────────────
st.set_page_config(page_title="Iris AI Explorer", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #fdfbfb; }
    
    /* Headers */
    .main-title { font-size: 3rem; font-weight: 800; color: #2d3436; text-align: center; margin-bottom: 0px; }
    .sub-title { font-size: 1.2rem; color: #636e72; text-align: center; margin-bottom: 40px; }
    
    /* Custom Prediction Card */
    .pred-card { background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; border: 1px solid #f1f2f6; }
    .pred-emoji { font-size: 4rem; margin-bottom: 10px; }
    .pred-species { font-size: 2.5rem; font-weight: 700; color: #6c5ce7; margin: 0; }
    .pred-conf { font-size: 1.2rem; color: #b2bec3; margin-top: 5px; }
    
    /* Button Styling */
    div.stButton > button:first-child { background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: white; border: none; border-radius: 10px; height: 50px; font-size: 18px; font-weight: 600; width: 100%; transition: all 0.3s ease; }
    div.stButton > button:first-child:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4); }
</style>
""", unsafe_allow_html=True)

# ── Data Loading & Model Training ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Iris (1).csv")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "predict_flower_species_combined (1).ipynb")

@st.cache_resource
def prepare_environment():
    # 1. Load Data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        # Fallback empty dataframe to prevent crash if file missing
        df = pd.DataFrame(columns=["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])
        return None, None, None, None, df, None

    # 2. Preprocessing
    le = LabelEncoder()
    df["Species_Encoded"] = le.fit_transform(df["Species"])
    
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df[features]
    Y = df["Species_Encoded"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    
    models["Logistic Regression"].fit(X_train_scaled, Y_train)
    models["KNN"].fit(X_train_scaled, Y_train)
    models["Naive Bayes"].fit(X_train, Y_train) # NB uses unscaled data
    
    # Calculate Accuracies
    accuracies = {
        "Logistic Regression": accuracy_score(Y_test, models["Logistic Regression"].predict(X_test_scaled)),
        "KNN": accuracy_score(Y_test, models["KNN"].predict(X_test_scaled)),
        "Naive Bayes": accuracy_score(Y_test, models["Naive Bayes"].predict(X_test))
    }
    
    return models, scaler, le, accuracies, df, features

models, scaler, le, accuracies, df, features = prepare_environment()

SPECIES_METADATA = {
    "Iris-setosa": {"emoji": "🌸", "color": "#ff7675", "desc": "Small, delicate petals. Usually easy to distinguish."},
    "Iris-versicolor": {"emoji": "🌺", "color": "#74b9ff", "desc": "Medium-sized features. Often falls in the middle range."},
    "Iris-virginica": {"emoji": "🌻", "color": "#55efc4", "desc": "The largest of the three with prominent petals and sepals."}
}

# ── App Header ──────────────────────────────────────────────────────────────
st.markdown("<p class='main-title'>Iris AI Explorer</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Machine Learning Classification for Botanical Identification</p>", unsafe_allow_html=True)

if df.empty:
    st.error(f"Dataset not found at `{CSV_PATH}`. Please ensure the file is in the same directory as this script.")
    st.stop()

# ── Main Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌸 AI Predictor", "📊 Dataset & Analytics", "💻 Architecture & Code"])

# ==============================================================================
# TAB 1: AI PREDICTOR
# ==============================================================================
with tab1:
    col_input, col_output = st.columns([1.2, 1], gap="large")
    
    with col_input:
        st.subheader("⚙️ Input Parameters", anchor=False)
        st.write("Adjust the slider values below to reflect the physical dimensions of the Iris flower.")
        
        with st.container(border=True):
            model_choice = st.selectbox("🧠 Select Classification Algorithm", ["Logistic Regression", "KNN", "Naive Bayes"], index=0)
            
            # Input Grid Layout
            c1, c2 = st.columns(2)
            with c1:
                sepal_len = st.slider("Sepal Length (cm)", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()), float(df["SepalLengthCm"].mean()), help="Length of the outer leaf.")
                petal_len = st.slider("Petal Length (cm)", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()), float(df["PetalLengthCm"].mean()), help="Length of the inner flower petal.")
            with c2:
                sepal_wid = st.slider("Sepal Width (cm)", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()), float(df["SepalWidthCm"].mean()), help="Width of the outer leaf.")
                petal_wid = st.slider("Petal Width (cm)", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()), float(df["PetalWidthCm"].mean()), help="Width of the inner flower petal.")
            
    with col_output:
        st.subheader("🎯 Real-Time Prediction", anchor=False)
        
        input_data = pd.DataFrame([[sepal_len, sepal_wid, petal_len, petal_wid]], columns=features)
        
        # Inference Logic
        if model_choice in ["Logistic Regression", "KNN"]:
            scaled_data = scaler.transform(input_data)
            pred_idx = models[model_choice].predict(scaled_data)[0]
            proba = models[model_choice].predict_proba(scaled_data)[0]
        else:
            pred_idx = models["Naive Bayes"].predict(input_data)[0]
            proba = models["Naive Bayes"].predict_proba(input_data)[0]
            
        species_name = le.inverse_transform([pred_idx])[0]
        confidence = proba[pred_idx] * 100
        meta = SPECIES_METADATA.get(species_name, {"emoji": "🌱", "color": "#000", "desc": ""})
        
        # Render Custom HTML Card
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-emoji">{meta['emoji']}</div>
            <p class="pred-species">{species_name.replace('Iris-', '').title()}</p>
            <p class="pred-conf">Confidence: <strong>{confidence:.1f}%</strong></p>
            <p style="color: #636e72; font-size: 0.9rem; margin-top: 15px;"><i>{meta['desc']}</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        
        # Plotly Probability Bar Chart
        prob_df = pd.DataFrame({"Species": le.classes_, "Probability": proba * 100})
        prob_df["Species"] = prob_df["Species"].str.replace("Iris-", "").str.title()
        
        fig = px.bar(prob_df, x="Probability", y="Species", orientation='h',
                     color="Species", color_discrete_map={"Setosa": "#ff7675", "Versicolor": "#74b9ff", "Virginica": "#55efc4"},
                     text_auto='.1f', range_x=[0, 100])
        fig.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=0, b=0), 
                          xaxis_title=None, yaxis_title=None, plot_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 2: DATASET & ANALYTICS
# ==============================================================================
with tab2:
    st.subheader("🔍 Exploratory Data Analysis")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total Samples", df.shape[0])
        st.metric("Features", len(features))
        st.metric("Target Classes", len(le.classes_))
        st.write("---")
        st.dataframe(df.drop(columns=["Species_Encoded"]).head(15), use_container_width=True, height=250)
        
        with open(CSV_PATH, "rb") as file:
             st.download_button(label="📥 Download Iris.csv", data=file, file_name="Iris.csv", mime="text/csv", use_container_width=True)

    with col2:
        # Interactive 3D Scatter Plot
        fig_3d = px.scatter_3d(df, x='SepalLengthCm', y='PetalLengthCm', z='PetalWidthCm',
                               color='Species', color_discrete_sequence=["#ff7675", "#74b9ff", "#55efc4"],
                               opacity=0.8, size_max=5)
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=450)
        st.plotly_chart(fig_3d, use_container_width=True)

# ==============================================================================
# TAB 3: ARCHITECTURE & CODE
# ==============================================================================
with tab3:
    st.subheader("📈 Model Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Logistic Regression Accuracy", f"{accuracies['Logistic Regression']*100:.1f}%")
    m2.metric("KNN Accuracy", f"{accuracies['KNN']*100:.1f}%")
    m3.metric("Naive Bayes Accuracy", f"{accuracies['Naive Bayes']*100:.1f}%")
    
    st.divider()
    
    col_code, col_dl = st.columns([3, 1])
    with col_code:
        st.subheader("Training Implementation (Python)")
        training_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load and Encode
df = pd.read_csv("Iris.csv")
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
        """
        st.code(training_code, language="python")
        
    with col_dl:
        st.subheader("Project Files")
        st.info("Download the original Jupyter Notebook containing the full Exploratory Data Analysis, model comparisons, and classification reports.")
        try:
            with open(NOTEBOOK_PATH, "rb") as nb_file:
                st.download_button(
                    label="📓 Download .ipynb",
                    data=nb_file,
                    file_name="predict_flower_species.ipynb",
                    mime="application/x-ipynb+json",
                    type="primary",
                    use_container_width=True
                )
        except FileNotFoundError:
            st.error("Jupyter Notebook file not found.")
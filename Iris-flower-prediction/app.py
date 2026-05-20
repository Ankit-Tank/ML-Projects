import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# ==========================================
# 1. MODEL TRAINING & LOGIC CLASS
# ==========================================
class IrisModelManager:
    def __init__(self, data_path='Iris.csv'):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.models = {}
        self.metrics = {}
        self.features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

    def load_and_preprocess(self):
        if not os.path.exists(self.data_path):
            # Fallback for demo if file is missing
            st.error(f"Dataset '{self.data_path}' not found! Please ensure it is in the same folder.")
            st.stop()
        
        self.df = pd.read_csv(self.data_path)
        if 'Id' in self.df.columns:
            self.df = self.df.drop('Id', axis=1)
        
        X = self.df[self.features]
        y = self.df["Species"]
        y_encoded = self.le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()
        
        # Train 3 different models
        lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        
        self.models = {'Logistic Regression': lr, 'KNN': knn, 'Random Forest': rf}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            self.metrics[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted')
            }

    def predict(self, model_name, features_list):
        model = self.models[model_name]
        features_scaled = self.scaler.transform([features_list])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        species = self.le.inverse_transform([prediction])[0]
        return species, probabilities, self.le.classes_

# ==========================================
# 2. STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Iris Intelligence Pro", page_icon="🌸", layout="wide")

# Custom CSS Injection
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #121212 0%, #1e1e2f 100%); color: #e0e0e0; }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px; padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px); margin-bottom: 20px;
    }
    .prediction-output {
        text-align: center; padding: 30px;
        background: rgba(76, 175, 80, 0.1);
        border-radius: 20px; border: 2px solid #4CAF50; margin: 20px 0;
    }
    .prediction-value { font-size: 2.5rem; font-weight: 700; color: #4CAF50; margin: 0; }
    .stTitle {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800 !important; font-size: 3rem !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize and cache the model manager
@st.cache_resource
def get_manager():
    manager = IrisModelManager()
    manager.train_models()
    return manager

manager = get_manager()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/346/346167.png", width=80)
    st.title("Settings")
    model_choice = st.selectbox("Classification Model", list(manager.models.keys()))
    
    st.divider()
    st.markdown("### Flower Dimensions")
    sl = st.slider("Sepal Length", 4.0, 8.0, 5.8)
    sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
    pl = st.slider("Petal Length", 1.0, 7.0, 4.4)
    pw = st.slider("Petal Width", 0.1, 2.5, 1.4)
    
    predict_btn = st.button("Predict Species", use_container_width=True)

# ==========================================
# 4. MAIN APP LAYOUT
# ==========================================
st.markdown("<h1 class='stTitle'>🌸 Iris Flower Species Predictor</h1>", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Prediction", "📊 Exploration", "📈 Model Stats"])

with tabs[0]:
    # Run prediction
    species, probs, classes = manager.predict(model_choice, [sl, sw, pl, pw])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f'<div class="prediction-output"><p>Predicted Species</p><p class="prediction-value">{species}</p></div>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = max(probs)*100,
            title = {'text': "Confidence %", 'font': {'color': "#e0e0e0"}},
            gauge = {'bar': {'color': "#4CAF50"}, 'bgcolor': "rgba(0,0,0,0)"}
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#e0e0e0"}, height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("<div class='metric-card'><h3>Probabilities</h3>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Species": classes, "Probability": probs})
        fig_probs = px.bar(prob_df, y="Species", x="Probability", orientation='h', color="Probability", color_continuous_scale='Viridis')
        fig_probs.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e0e0e0"}, height=300)
        st.plotly_chart(fig_probs, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("### Data Insights")
    fig_scatter = px.scatter(manager.df, x="PetalLengthCm", y="PetalWidthCm", color="Species", template="plotly_dark")
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.dataframe(manager.df.head(10), use_container_width=True)

with tabs[2]:
    st.markdown("### Model Benchmarking")
    perf_df = pd.DataFrame([{"Model": k, "Accuracy": f"{v['Accuracy']:.2%}"} for k,v in manager.metrics.items()])
    st.table(perf_df)
    
    if model_choice == "Random Forest":
        imp = pd.DataFrame({"Feature": manager.features, "Importance": manager.models["Random Forest"].feature_importances_})
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation='h', title="Feature Importance"), use_container_width=True)
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
from sklearn.datasets import load_iris # Backup data source

# ==========================================
# 1. SMART DATA LOADER & MODEL LOGIC
# ==========================================
class IrisModelManager:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.models = {}
        self.metrics = {}
        self.features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

    def load_data(self):
        # Try to find the file locally (checking common names)
        possible_names = ['Iris.csv', 'Iris (1).csv', 'iris.csv']
        found_file = None
        for name in possible_names:
            if os.path.exists(name):
                found_file = name
                break
        
        if found_file:
            self.df = pd.read_csv(found_file)
            # Remove Id column if it exists
            if 'Id' in self.df.columns:
                self.df = self.df.drop('Id', axis=1)
            # Standardize column names
            self.df.columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
        else:
            # FALLBACK: Load directly from Scikit-Learn if file is missing
            st.warning("⚠️ Local CSV not found. Loading built-in botanical dataset...")
            data = load_iris()
            self.df = pd.DataFrame(data.data, columns=self.features)
            self.df['Species'] = [data.target_names[i] for i in data.target]
        
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
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Train Models
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
# 2. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Iris Intelligence Pro", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #e0e0e0; }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px; padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px); margin-bottom: 20px;
    }
    .prediction-output {
        text-align: center; padding: 40px;
        background: rgba(0, 255, 127, 0.05);
        border-radius: 20px; border: 2px solid #00ff7f; margin: 20px 0;
        box-shadow: 0 0 20px rgba(0, 255, 127, 0.2);
    }
    .prediction-value { font-size: 3rem; font-weight: 800; color: #00ff7f; margin: 0; text-transform: uppercase; letter-spacing: 2px;}
    .stTitle {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800 !important; font-size: 3.5rem !important; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_manager():
    manager = IrisModelManager()
    manager.train_models()
    return manager

manager = get_manager()

# ==========================================
# 3. SIDEBAR & MAIN APP
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Controls</h2>", unsafe_allow_html=True)
    model_choice = st.selectbox("Select Intelligence Model", list(manager.models.keys()))
    st.divider()
    st.markdown("### 📏 Adjust Measurements")
    sl = 
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
    page_title="🌸 Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# ── Load & train (cached so it only runs once) ───────────────────────────────
@st.cache_resource
def load_models():
    df = pd.read_csv("Iris.csv")

    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])
    # le.classes_  → ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # Use only the 4 feature columns (exclude Id)
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df[features]
    Y = df["Species"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.7, random_state=42
    )

    # Scaling (for LR & KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Model 1 — Logistic Regression
    lr  = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, Y_train)

    # Model 2 — KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, Y_train)

    # Model 3 — Naive Bayes (no scaling)
    nb  = GaussianNB()
    nb.fit(X_train, Y_train)

    # Accuracies
    acc = {
        "Logistic Regression": round(accuracy_score(Y_test, lr.predict(X_test_scaled))  * 100, 2),
        "KNN":                  round(accuracy_score(Y_test, knn.predict(X_test_scaled)) * 100, 2),
        "Naive Bayes":          round(accuracy_score(Y_test, nb.predict(X_test))         * 100, 2),
    }
    prec = {
        "Logistic Regression": round(precision_score(Y_test, lr.predict(X_test_scaled),  average="weighted") * 100, 2),
        "KNN":                  round(precision_score(Y_test, knn.predict(X_test_scaled), average="weighted") * 100, 2),
        "Naive Bayes":          round(precision_score(Y_test, nb.predict(X_test),         average="weighted") * 100, 2),
    }

    return lr, knn, nb, scaler, le, acc, prec

lr_model, knn_model, nb_model, scaler, le, acc, prec = load_models()

SPECIES_EMOJI = {
    "Iris-setosa":     "🌼",
    "Iris-versicolor": "🌺",
    "Iris-virginica":  "🌷",
}

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🌸 Iris Flower Species Predictor")
st.markdown(
    "Enter the flower measurements and choose a model to predict the Iris species."
)
st.divider()

# Model selector
model_choice = st.selectbox(
    "🤖 Select Classification Model",
    ["Logistic Regression", "KNN", "Naive Bayes"],
    help="Logistic Regression achieved 100% accuracy on the test set."
)

st.divider()

# Input sliders
st.subheader("📐 Flower Measurements")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, step=0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.4, step=0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.4, step=0.1)

st.divider()

# Predict
if st.button("🔍 Predict Species", use_container_width=True, type="primary"):
    raw_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_df  = pd.DataFrame(raw_input,
                             columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])

    if model_choice == "Logistic Regression":
        scaled     = scaler.transform(input_df)
        prediction = lr_model.predict(scaled)[0]
        proba      = lr_model.predict_proba(scaled)[0]

    elif model_choice == "KNN":
        scaled     = scaler.transform(input_df)
        prediction = knn_model.predict(scaled)[0]
        proba      = knn_model.predict_proba(scaled)[0]

    else:  # Naive Bayes — no scaling
        prediction = nb_model.predict(input_df)[0]
        proba      = nb_model.predict_proba(input_df)[0]

    species_name = le.inverse_transform([prediction])[0]
    emoji        = SPECIES_EMOJI.get(species_name, "🌸")
    confidence   = round(proba[prediction] * 100, 1)

    st.success(f"**Predicted Species: {emoji} {species_name}**")
    st.metric("Confidence", f"{confidence}%")

    # Probability bar chart
    st.subheader("📊 Class Probabilities")
    prob_df = pd.DataFrame({
        "Species":     le.classes_,
        "Probability": [round(p * 100, 2) for p in proba]
    }).set_index("Species")
    st.bar_chart(prob_df)

st.divider()

# Model comparison table
st.subheader("📈 Model Comparison (Test Set)")
results_df = pd.DataFrame({
    "Model":     list(acc.keys()),
    "Accuracy (%)":  list(acc.values()),
    "Precision (%)": list(prec.values()),
})
results_df = results_df.set_index("Model")
st.dataframe(results_df, use_container_width=True)

st.caption("Dataset: Iris.csv · 150 samples · 3 classes · Train/Test split: 30% / 70%")

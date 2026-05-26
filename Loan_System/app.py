import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise · Loan Intelligence System",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (injected once)
# ─────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Root Variables ───────────────────────── */
:root {
  --bg-base:    #06090F;
  --bg-card:    #0D1220;
  --bg-card2:   #111827;
  --border:     rgba(255,255,255,0.07);
  --gold:       #C8A96E;
  --gold-light: #E2C98A;
  --gold-dim:   rgba(200,169,110,0.15);
  --emerald:    #10B981;
  --crimson:    #EF4444;
  --sky:        #38BDF8;
  --text-1:     #F0EDE8;
  --text-2:     #9CA3AF;
  --text-3:     #4B5563;
}

/* ── Wipe Streamlit defaults ──────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg-base) !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-1) !important;
}
[data-testid="stSidebar"] {
    background: #080C17 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1400px; }

/* ── Hide Streamlit chrome ────────────────── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

/* ── Typography ───────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
}

/* ── Sidebar inputs ───────────────────────── */
[data-testid="stSidebar"] label {
    color: var(--text-2) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #0D1220 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] [data-baseweb="select"]:hover {
    border-color: var(--gold) !important;
}

/* ── Sliders ──────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--gold) !important;
    border: 2px solid #0D1220 !important;
}

/* ── Buttons ──────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] > button,
[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #C8A96E 0%, #9A7A42 100%) !important;
    color: #060910 !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
[data-testid="stSidebar"] button:hover {
    box-shadow: 0 0 24px rgba(200,169,110,0.4) !important;
    transform: translateY(-1px) !important;
}

/* main area button */
[data-testid="stBaseButton-primary"] > button {
    background: linear-gradient(135deg, #C8A96E 0%, #9A7A42 100%) !important;
    color: #060910 !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
}

/* ── Number input ─────────────────────────── */
[data-testid="stNumberInput"] input { 
    background: #0D1220 !important; 
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: var(--text-1) !important;
    border-radius: 8px !important;
}

/* ── Select boxes ─────────────────────────── */
[data-baseweb="select"] > div {
    background: #0D1220 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
[data-baseweb="popover"] { background: #0D1220 !important; }

/* ── Section divider ──────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Tabs ─────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    color: var(--text-2) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: transparent !important;
}

/* ── Dataframe / table ────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: HTML COMPONENTS
# ─────────────────────────────────────────────
def hero_header():
    st.markdown("""
    <div style="
        display:flex; align-items:center; gap:18px;
        padding: 2.2rem 0 1.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 2rem;
    ">
        <div style="
            width:54px; height:54px; border-radius:14px;
            background: linear-gradient(135deg,#C8A96E,#7A5C2E);
            display:flex; align-items:center; justify-content:center;
            font-size:26px; flex-shrink:0;
            box-shadow: 0 0 30px rgba(200,169,110,0.3);
        ">⬡</div>
        <div>
            <div style="
                font-family:'Cormorant Garamond',serif;
                font-size:2.05rem; font-weight:700;
                color:#F0EDE8; line-height:1;
                letter-spacing:0.02em;
            ">CreditWise</div>
            <div style="
                color:#9CA3AF; font-size:0.82rem;
                font-family:'Outfit',sans-serif;
                letter-spacing:0.12em; text-transform:uppercase;
                margin-top:4px;
            ">SecureTrust Bank · Intelligent Loan Analysis System</div>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="
                background: rgba(16,185,129,0.12);
                border:1px solid rgba(16,185,129,0.3);
                border-radius:20px; padding:5px 14px;
                font-size:0.72rem; color:#10B981;
                font-family:'Outfit',sans-serif;
                letter-spacing:0.08em; text-transform:uppercase;
                font-weight:600;
            ">● ML Engine Online</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label, value, color="#C8A96E", sub=None):
    sub_html = f'<div style="color:#6B7280;font-size:0.72rem;margin-top:2px;font-family:Outfit,sans-serif">{sub}</div>' if sub else ""
    return f"""
    <div style="
        background:#0D1220; border:1px solid rgba(255,255,255,0.07);
        border-radius:14px; padding:1.2rem 1.4rem;
        border-top: 2px solid {color}40;
    ">
        <div style="color:#6B7280;font-size:0.72rem;letter-spacing:0.1em;
            text-transform:uppercase;font-family:Outfit,sans-serif;margin-bottom:6px">{label}</div>
        <div style="color:{color};font-family:'Cormorant Garamond',serif;
            font-size:2rem;font-weight:700;line-height:1">{value}</div>
        {sub_html}
    </div>"""


def section_title(text, sub=None):
    sub_html = f'<div style="color:#6B7280;font-size:0.82rem;font-family:Outfit,sans-serif;margin-top:4px">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="margin:2rem 0 1.2rem 0;">
        <div style="font-family:\'Cormorant Garamond\',serif;font-size:1.5rem;
            font-weight:700;color:#F0EDE8;">{text}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def result_card_approved(confidence, model_name):
    pct = int(confidence * 100)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,rgba(16,185,129,0.08) 0%,rgba(6,9,15,0) 60%);
        border:1px solid rgba(16,185,129,0.35); border-radius:18px;
        padding:2.4rem 2rem; text-align:center; margin:1rem 0;
        box-shadow: 0 0 60px rgba(16,185,129,0.08);
    ">
        <div style="font-size:3rem;margin-bottom:8px">✅</div>
        <div style="font-family:\'Cormorant Garamond\',serif;font-size:2.8rem;
            font-weight:700;color:#10B981;margin-bottom:6px">Loan Approved</div>
        <div style="color:#6EE7B7;font-family:Outfit,sans-serif;font-size:0.9rem;
            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:1.5rem">
            Application meets approval criteria
        </div>
        <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
            <div>
                <div style="color:#6B7280;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;font-family:Outfit,sans-serif">Confidence</div>
                <div style="color:#10B981;font-family:\'Cormorant Garamond\',serif;
                    font-size:2.2rem;font-weight:700">{pct}%</div>
            </div>
            <div>
                <div style="color:#6B7280;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;font-family:Outfit,sans-serif">Model Used</div>
                <div style="color:#F0EDE8;font-family:\'Cormorant Garamond\',serif;
                    font-size:1.5rem;font-weight:600">{model_name}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def result_card_rejected(confidence, model_name):
    pct = int(confidence * 100)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,rgba(239,68,68,0.08) 0%,rgba(6,9,15,0) 60%);
        border:1px solid rgba(239,68,68,0.35); border-radius:18px;
        padding:2.4rem 2rem; text-align:center; margin:1rem 0;
        box-shadow: 0 0 60px rgba(239,68,68,0.06);
    ">
        <div style="font-size:3rem;margin-bottom:8px">❌</div>
        <div style="font-family:\'Cormorant Garamond\',serif;font-size:2.8rem;
            font-weight:700;color:#EF4444;margin-bottom:6px">Loan Rejected</div>
        <div style="color:#FCA5A5;font-family:Outfit,sans-serif;font-size:0.9rem;
            letter-spacing:0.08em;text-transform:uppercase;margin-bottom:1.5rem">
            Application does not meet approval criteria
        </div>
        <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
            <div>
                <div style="color:#6B7280;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;font-family:Outfit,sans-serif">Confidence</div>
                <div style="color:#EF4444;font-family:\'Cormorant Garamond\',serif;
                    font-size:2.2rem;font-weight:700">{pct}%</div>
            </div>
            <div>
                <div style="color:#6B7280;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;font-family:Outfit,sans-serif">Model Used</div>
                <div style="color:#F0EDE8;font-family:\'Cormorant Garamond\',serif;
                    font-size:1.5rem;font-weight:600">{model_name}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ML PIPELINE  (exact replica of notebook)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    df = pd.read_csv("loan_approval_data.csv")

    # Identify types
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols   = df.select_dtypes(include=["number"]).columns

    # ── Imputation ────────────────────────────
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # ── Drop ID ───────────────────────────────
    df = df.drop("Applicant_ID", axis=1)

    # ── Label Encoding ────────────────────────
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    edu_classes = le.classes_.copy()          # save: ['Graduate','Not Graduate']

    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])   # No=0, Yes=1

    # ── One-Hot Encoding ──────────────────────
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded    = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols),
                              index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # ── Feature Engineering ───────────────────
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    # ── Features & Target ─────────────────────
    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    Y = df["Loan_Approved"]

    feature_names = list(X.columns)

    # ── Train / Test Split ────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # ── Scaling ───────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Train Models ──────────────────────────
    log_model = LogisticRegression()
    log_model.fit(X_train_scaled, Y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, Y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, Y_train)

    models = {
        "Logistic Regression": log_model,
        "KNN (k=5)":           knn_model,
        "Naive Bayes":         nb_model,
    }

    # ── Evaluate ──────────────────────────────
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        metrics[name] = {
            "accuracy":  accuracy_score(Y_test, y_pred),
            "precision": precision_score(Y_test, y_pred),
            "recall":    recall_score(Y_test, y_pred),
            "f1":        f1_score(Y_test, y_pred),
            "cm":        confusion_matrix(Y_test, y_pred),
        }

    return {
        "models":        models,
        "scaler":        scaler,
        "ohe":           ohe,
        "ohe_cols":      ohe_cols,
        "edu_classes":   edu_classes,
        "feature_names": feature_names,
        "metrics":       metrics,
        "X_test":        pd.DataFrame(X_test_scaled, columns=feature_names),
        "Y_test":        Y_test.values,
    }


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_applicant(pipeline, inputs, model_name):
    """
    Takes raw user inputs, replicates the notebook preprocessing,
    and returns (prediction_label, probability_dict).
    """
    ohe          = pipeline["ohe"]
    ohe_cols     = pipeline["ohe_cols"]
    scaler       = pipeline["scaler"]
    edu_classes  = pipeline["edu_classes"]
    feature_names= pipeline["feature_names"]
    model        = pipeline["models"][model_name]

    # ── Encode Education_Level ────────────────
    edu_encoded = int(np.where(edu_classes == inputs["Education_Level"])[0][0])

    # ── Encode categorical via OHE ────────────
    cat_df = pd.DataFrame([[
        inputs["Employment_Status"],
        inputs["Marital_Status"],
        inputs["Loan_Purpose"],
        inputs["Property_Area"],
        inputs["Gender"],
        inputs["Employer_Category"],
    ]], columns=ohe_cols)
    ohe_encoded = ohe.transform(cat_df)
    ohe_df      = pd.DataFrame(ohe_encoded,
                               columns=ohe.get_feature_names_out(ohe_cols))

    # ── Feature Engineering ───────────────────
    dti_sq    = inputs["DTI_Ratio"] ** 2
    credit_sq = inputs["Credit_Score"] ** 2

    # ── Assemble numeric base row ─────────────
    base = {
        "Applicant_Income":   inputs["Applicant_Income"],
        "Coapplicant_Income": inputs["Coapplicant_Income"],
        "Age":                inputs["Age"],
        "Dependents":         inputs["Dependents"],
        "Existing_Loans":     inputs["Existing_Loans"],
        "Savings":            inputs["Savings"],
        "Collateral_Value":   inputs["Collateral_Value"],
        "Loan_Amount":        inputs["Loan_Amount"],
        "Loan_Term":          inputs["Loan_Term"],
        "Education_Level":    edu_encoded,
    }
    row = pd.DataFrame([base])
    row = pd.concat([row.reset_index(drop=True),
                     ohe_df.reset_index(drop=True)], axis=1)
    row["DTI_Ratio_sq"]    = dti_sq
    row["Credit_Score_sq"] = credit_sq

    # ── Reindex to match training feature order ─
    row = row.reindex(columns=feature_names, fill_value=0.0)

    # ── Scale ─────────────────────────────────
    row_scaled = scaler.transform(row)

    # ── Predict ───────────────────────────────
    pred        = model.predict(row_scaled)[0]
    proba       = model.predict_proba(row_scaled)[0]
    label       = "Approved" if pred == 1 else "Rejected"
    confidence  = proba[pred]

    return label, confidence, proba


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#9CA3AF"),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
)


def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "family": "Cormorant Garamond",
                                         "color": color}},
        title={"text": title, "font": {"size": 13, "family": "Outfit", "color": "#6B7280"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151",
                     "tickfont": {"size": 10}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#0D1220",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.08)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.08)"},
                {"range": [75, 100],"color": "rgba(16,185,129,0.08)"},
            ],
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200)
    return fig


def make_cm_heatmap(cm, model_name):
    labels = ["Rejected", "Approved"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=cm, texttemplate="%{text}",
        colorscale=[[0, "#0D1220"], [0.5, "#9A7A42"], [1, "#C8A96E"]],
        showscale=False,
        textfont={"size": 20, "family": "Cormorant Garamond", "color": "#F0EDE8"},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#9CA3AF"),
        margin=dict(l=0, r=0, t=30, b=0),

        title=dict(text=model_name, font=dict(size=14, color="#9CA3AF")),

        xaxis=dict(
            title="Predicted",
            side="bottom",
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor="rgba(0,0,0,0)"
        ),

        yaxis=dict(
            title="Actual",
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor="rgba(0,0,0,0)"
        ),

        height=260,
    )

    return fig


def make_bar_comparison(metrics):
    model_names = list(metrics.keys())
    acc  = [metrics[m]["accuracy"]  for m in model_names]
    prec = [metrics[m]["precision"] for m in model_names]
    rec  = [metrics[m]["recall"]    for m in model_names]
    f1   = [metrics[m]["f1"]        for m in model_names]

    fig = go.Figure()
    palette = ["#C8A96E", "#38BDF8", "#10B981", "#A78BFA"]

    for vals, label, clr in zip(
        [acc, prec, rec, f1],
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        palette,
    ):
        fig.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=[v * 100 for v in vals],
            marker_color=clr,
            marker_line_color="rgba(0,0,0,0)",
            opacity=0.9,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#9CA3AF"),
        margin=dict(l=0, r=0, t=30, b=0),

        barmode="group",

        yaxis=dict(
            title="Score (%)",
            range=[0, 105],
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.05)"
        ),

        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor="rgba(0,0,0,0)"
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CA3AF")
        ),

        height=340,
    )

    return fig


def make_proba_donut(proba, labels=("Rejected", "Approved")):
    colors = ["#EF4444", "#10B981"]
    fig = go.Figure(go.Pie(
        values=[proba[0]*100, proba[1]*100],
        labels=list(labels),
        hole=0.65,
        marker=dict(colors=colors,
                    line=dict(color="#06090F", width=3)),
        textinfo="label+percent",
        textfont=dict(family="Outfit", size=12, color="#F0EDE8"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                      showlegend=False,
                      annotations=[dict(
                          text=f"{max(proba)*100:.0f}%",
                          font=dict(size=28, family="Cormorant Garamond",
                                    color="#C8A96E"),
                          showarrow=False,
                      )])
    return fig


# ─────────────────────────────────────────────
# SIDEBAR – APPLICATION FORM
# ─────────────────────────────────────────────
def render_sidebar(pipeline):
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.4rem 0 1rem 0;border-bottom:1px solid rgba(255,255,255,0.06)">
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.35rem;
                font-weight:700;color:#F0EDE8;">Loan Application</div>
            <div style="font-size:0.75rem;color:#6B7280;margin-top:3px;
                font-family:Outfit,sans-serif;letter-spacing:0.05em">
                Fill in applicant details below
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Model Selector
        model_name = st.selectbox(
            "🤖  Select Model",
            ["Logistic Regression", "KNN (k=5)", "Naive Bayes"],
            help="Choose which trained model to use for prediction",
        )

        st.markdown("---")
        st.markdown("""<div style="font-size:0.7rem;color:#4B5563;
            letter-spacing:0.1em;text-transform:uppercase;
            font-family:Outfit,sans-serif;margin-bottom:8px">
            Personal Information</div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80,
                                  value=35, step=1)
        with col2:
            dependents = st.number_input("Dependents", min_value=0,
                                         max_value=10, value=1, step=1)

        gender = st.selectbox("Gender", ["Female", "Male"])
        marital = st.selectbox("Marital Status", ["Married", "Single"])
        education = st.selectbox("Education Level",
                                 ["Graduate", "Not Graduate"])

        st.markdown("---")
        st.markdown("""<div style="font-size:0.7rem;color:#4B5563;
            letter-spacing:0.1em;text-transform:uppercase;
            font-family:Outfit,sans-serif;margin-bottom:8px">
            Employment & Income</div>""", unsafe_allow_html=True)

        employment = st.selectbox("Employment Status",
                                  ["Salaried", "Self-employed",
                                   "Contract", "Unemployed"])
        employer_cat = st.selectbox("Employer Category",
                                    ["Business", "Government", "MNC",
                                     "Private", "Unemployed"])

        applicant_income  = st.number_input("Applicant Income (₹)",
                                            min_value=0, value=8000, step=100)
        coapplicant_income= st.number_input("Co-applicant Income (₹)",
                                             min_value=0, value=2000, step=100)

        st.markdown("---")
        st.markdown("""<div style="font-size:0.7rem;color:#4B5563;
            letter-spacing:0.1em;text-transform:uppercase;
            font-family:Outfit,sans-serif;margin-bottom:8px">
            Financial Profile</div>""", unsafe_allow_html=True)

        credit_score = st.slider("Credit Score", 300, 900, 680)
        dti_ratio    = st.slider("DTI Ratio", 0.0, 1.0, 0.30, 0.01,
                                 help="Debt-to-Income ratio (0 = no debt)")
        existing_loans= st.number_input("Existing Loans", min_value=0,
                                         max_value=10, value=1, step=1)
        savings       = st.number_input("Savings Balance (₹)",
                                         min_value=0, value=10000, step=500)
        collateral    = st.number_input("Collateral Value (₹)",
                                         min_value=0, value=20000, step=500)

        st.markdown("---")
        st.markdown("""<div style="font-size:0.7rem;color:#4B5563;
            letter-spacing:0.1em;text-transform:uppercase;
            font-family:Outfit,sans-serif;margin-bottom:8px">
            Loan Details</div>""", unsafe_allow_html=True)

        loan_amount = st.number_input("Loan Amount (₹)",
                                      min_value=1000, value=15000, step=500)
        loan_term   = st.selectbox("Loan Term (months)",
                                   [12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 360])
        loan_purpose = st.selectbox("Loan Purpose",
                                    ["Business", "Car", "Education",
                                     "Home", "Personal"])
        property_area= st.selectbox("Property Area",
                                    ["Rural", "Semiurban", "Urban"])

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        analyze = st.button("⬡  Analyze Application", use_container_width=True)

    inputs = {
        "Age":                age,
        "Dependents":         float(dependents),
        "Gender":             gender,
        "Marital_Status":     marital,
        "Education_Level":    education,
        "Employment_Status":  employment,
        "Employer_Category":  employer_cat,
        "Applicant_Income":   float(applicant_income),
        "Coapplicant_Income": float(coapplicant_income),
        "Credit_Score":       float(credit_score),
        "DTI_Ratio":          float(dti_ratio),
        "Existing_Loans":     float(existing_loans),
        "Savings":            float(savings),
        "Collateral_Value":   float(collateral),
        "Loan_Amount":        float(loan_amount),
        "Loan_Term":          float(loan_term),
        "Loan_Purpose":       loan_purpose,
        "Property_Area":      property_area,
    }
    return inputs, model_name, analyze


# ─────────────────────────────────────────────
# MAIN  APP
# ─────────────────────────────────────────────
def main():
    # Load & train (cached)
    with st.spinner("Initialising ML engine…"):
        pipeline = load_and_train()

    metrics = pipeline["metrics"]

    # Render sidebar form
    inputs, model_name, analyze = render_sidebar(pipeline)

    # ── Hero ──────────────────────────────────
    hero_header()

    # ── Overview stat cards ───────────────────
    best_model = max(metrics, key=lambda m: metrics[m]["accuracy"])
    best_acc   = metrics[best_model]["accuracy"]
    lr_acc     = metrics["Logistic Regression"]["accuracy"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(stat_card("Best Model Accuracy", f"{best_acc*100:.1f}%",
                              "#C8A96E", best_model), unsafe_allow_html=True)
    with col2:
        st.markdown(stat_card("Logistic Regression", f"{lr_acc*100:.1f}%",
                              "#38BDF8", f"Accuracy"), unsafe_allow_html=True)
    with col3:
        knn_acc = metrics["KNN (k=5)"]["accuracy"]
        st.markdown(stat_card("KNN  (k=5)", f"{knn_acc*100:.1f}%",
                              "#A78BFA", "Accuracy"), unsafe_allow_html=True)
    with col4:
        nb_acc = metrics["Naive Bayes"]["accuracy"]
        st.markdown(stat_card("Naive Bayes", f"{nb_acc*100:.1f}%",
                              "#10B981", "Accuracy"), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["  Prediction  ",
                                 "  Model Performance  ",
                                 "  Confusion Matrices  "])

    with tab1:
        if not analyze:
            st.markdown("""
            <div style="
                background:#0D1220;border:1px solid rgba(255,255,255,0.07);
                border-radius:16px;padding:3rem;text-align:center;margin-top:1rem;
            ">
                <div style="font-size:2.5rem;margin-bottom:12px">⬡</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:1.6rem;
                    font-weight:600;color:#F0EDE8;margin-bottom:8px">
                    Ready to Analyse
                </div>
                <div style="color:#4B5563;font-family:Outfit,sans-serif;font-size:0.88rem">
                    Complete the application form in the sidebar and click<br>
                    <strong style="color:#C8A96E">⬡ Analyse Application</strong> to get the prediction.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            label, conf, proba = predict_applicant(pipeline, inputs, model_name)

            if label == "Approved":
                result_card_approved(conf, model_name)
            else:
                result_card_rejected(conf, model_name)

            # Probability donut + gauge side by side
            c1, c2 = st.columns([1, 1])
            with c1:
                section_title("Probability Distribution",
                              "Rejection vs Approval confidence split")
                st.plotly_chart(make_proba_donut(proba),
                                use_container_width=True, config={"displayModeBar": False})
            with c2:
                section_title("Decision Metrics",
                              "Key financial indicators of this application")

                g1, g2 = st.columns(2)
                with g1:
                    color = "#10B981" if inputs["Credit_Score"] >= 650 else "#EF4444"
                    st.plotly_chart(
                        make_gauge(inputs["Credit_Score"] / 900,
                                   "Credit Score", color),
                        use_container_width=True, config={"displayModeBar": False})
                with g2:
                    color = "#10B981" if inputs["DTI_Ratio"] <= 0.40 else "#EF4444"
                    st.plotly_chart(
                        make_gauge(inputs["DTI_Ratio"],
                                   "DTI Ratio", color),
                        use_container_width=True, config={"displayModeBar": False})

                # Applicant quick summary
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.06);
                    border-radius:12px;padding:1.1rem 1.3rem;margin-top:4px">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                        <div>
                            <span style="color:#4B5563;font-size:0.72rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">Income</span>
                            <div style="color:#F0EDE8;font-family:'Cormorant Garamond',serif;
                                font-size:1.2rem;font-weight:600">
                                ₹{int(inputs['Applicant_Income']):,}</div>
                        </div>
                        <div>
                            <span style="color:#4B5563;font-size:0.72rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">Loan Ask</span>
                            <div style="color:#F0EDE8;font-family:'Cormorant Garamond',serif;
                                font-size:1.2rem;font-weight:600">
                                ₹{int(inputs['Loan_Amount']):,}</div>
                        </div>
                        <div>
                            <span style="color:#4B5563;font-size:0.72rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">Savings</span>
                            <div style="color:#F0EDE8;font-family:'Cormorant Garamond',serif;
                                font-size:1.2rem;font-weight:600">
                                ₹{int(inputs['Savings']):,}</div>
                        </div>
                        <div>
                            <span style="color:#4B5563;font-size:0.72rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">Collateral</span>
                            <div style="color:#F0EDE8;font-family:'Cormorant Garamond',serif;
                                font-size:1.2rem;font-weight:600">
                                ₹{int(inputs['Collateral_Value']):,}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        section_title("Model Comparison",
                      "Side-by-side evaluation across all three classifiers")

        st.plotly_chart(make_bar_comparison(metrics),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Detailed metrics table
        rows = []
        for m, v in metrics.items():
            rows.append({
                "Model": m,
                "Accuracy":  f"{v['accuracy']*100:.2f}%",
                "Precision": f"{v['precision']*100:.2f}%",
                "Recall":    f"{v['recall']*100:.2f}%",
                "F1 Score":  f"{v['f1']*100:.2f}%",
            })
        tbl = pd.DataFrame(rows).set_index("Model")

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] table {
            font-family:Outfit,sans-serif !important;
        }
        </style>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 3, 1])
        with c2:
            st.dataframe(tbl, use_container_width=True)

        # Individual gauges
        section_title("Per-model Accuracy Gauges", "")
        g1, g2, g3 = st.columns(3)
        colors = {"Logistic Regression": "#C8A96E",
                  "KNN (k=5)":           "#A78BFA",
                  "Naive Bayes":         "#10B981"}
        for col, (mname, mvals) in zip([g1, g2, g3], metrics.items()):
            with col:
                st.plotly_chart(
                    make_gauge(mvals["accuracy"], mname, colors[mname]),
                    use_container_width=True, config={"displayModeBar": False})

    with tab3:
        section_title("Confusion Matrices",
                      "True vs predicted labels on the 20% test set")

        c1, c2, c3 = st.columns(3)
        for col, (mname, mvals) in zip([c1, c2, c3], metrics.items()):
            with col:
                st.plotly_chart(
                    make_cm_heatmap(mvals["cm"], mname),
                    use_container_width=True, config={"displayModeBar": False})

                # Per-model stat row
                st.markdown(f"""
                <div style="background:#0D1220;border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;padding:1rem;margin-top:-8px">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                        <div style="text-align:center">
                            <div style="color:#4B5563;font-size:0.65rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">Precision</div>
                            <div style="color:#C8A96E;font-family:'Cormorant Garamond',serif;
                                font-size:1.3rem;font-weight:600">
                                {mvals['precision']*100:.1f}%</div>
                        </div>
                        <div style="text-align:center">
                            <div style="color:#4B5563;font-size:0.65rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">Recall</div>
                            <div style="color:#38BDF8;font-family:'Cormorant Garamond',serif;
                                font-size:1.3rem;font-weight:600">
                                {mvals['recall']*100:.1f}%</div>
                        </div>
                        <div style="text-align:center">
                            <div style="color:#4B5563;font-size:0.65rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">F1 Score</div>
                            <div style="color:#10B981;font-family:'Cormorant Garamond',serif;
                                font-size:1.3rem;font-weight:600">
                                {mvals['f1']*100:.1f}%</div>
                        </div>
                        <div style="text-align:center">
                            <div style="color:#4B5563;font-size:0.65rem;
                                text-transform:uppercase;letter-spacing:0.08em;
                                font-family:Outfit">Accuracy</div>
                            <div style="color:#A78BFA;font-family:'Cormorant Garamond',serif;
                                font-size:1.3rem;font-weight:600">
                                {mvals['accuracy']*100:.1f}%</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Footer ──────────────────────────────────
    st.markdown("""
    <div style="
        margin-top:4rem; padding:1.4rem 0;
        border-top:1px solid rgba(255,255,255,0.05);
        text-align:center; color:#374151;
        font-size:0.75rem; font-family:Outfit,sans-serif;
        letter-spacing:0.04em;
    ">
        CreditWise · SecureTrust Bank ML Loan System ·
        Models: Logistic Regression · KNN · Naive Bayes
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

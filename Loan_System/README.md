# ⬡ CreditWise — Intelligent Loan Analysis System
**SecureTrust Bank · ML-powered Loan Approval Dashboard**

---

## Overview

CreditWise is a production-grade Streamlit dashboard that wraps the SecureTrust Bank loan approval ML pipeline. It preserves the original notebook's preprocessing and model logic while delivering a modern, premium dark-theme UI with interactive Plotly charts.

**Three classifiers trained and compared:**
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | **87.5%** | 79.0% | 80.3% | 79.7% |
| Naive Bayes | 86.5% | 78.3% | 77.0% | 77.7% |
| KNN (k=5) | 75.5% | 62.0% | 50.8% | 55.9% |

---

## ML Pipeline (unchanged from notebook)

1. Load `loan_approval_data.csv` (1,000 records, 20 features)
2. Impute — mean for numericals, most-frequent for categoricals
3. Drop `Applicant_ID`
4. `LabelEncoder` → `Education_Level`, `Loan_Approved`
5. `OneHotEncoder(drop='first')` → 6 categorical columns → 15 binary features
6. Feature engineering → `DTI_Ratio²`, `Credit_Score²`; drop raw `DTI_Ratio`, `Credit_Score`
7. `train_test_split(test_size=0.2, random_state=42)`
8. `StandardScaler` (fit on train only)
9. Train: `LogisticRegression()`, `KNeighborsClassifier(n_neighbors=5)`, `GaussianNB()`

---

## Project Structure

```
creditwise/
├── app.py                    # Streamlit application (main entry point)
├── loan_approval_data.csv    # Dataset
├── requirements.txt          # Python dependencies
├── netlify.toml              # Netlify config (for static landing page)
├── .streamlit/
│   └── config.toml           # Dark theme + server config
└── public/
    └── index.html            # Static marketing/landing page → Netlify
```

---

## Local Development

```bash
# 1. Clone / unzip the project
cd creditwise

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Deployment

### Streamlit App → Streamlit Community Cloud (free)

1. Push the project to a **GitHub repository** (make sure `loan_approval_data.csv` is included).
2. Go to **[https://streamlit.io/cloud](https://streamlit.io/cloud)** and sign in with GitHub.
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`.
4. Click **Deploy**. Your app URL will be `https://your-username-creditwise.streamlit.app`.

> Alternative free hosts: **Render**, **Railway**, **Hugging Face Spaces** (Streamlit SDK)

### Landing Page → Netlify (static)

1. The `public/` folder contains a standalone static HTML landing page.
2. After deploying the Streamlit app, **replace all `https://your-app.streamlit.app`** URLs in `public/index.html` with your actual Streamlit Cloud URL.
3. Connect your GitHub repo to **[https://netlify.com](https://netlify.com)**.
4. Build settings:
   - **Build command:** *(leave blank — no build needed)*
   - **Publish directory:** `public`
5. Deploy. Your landing page will be live on a `*.netlify.app` domain.

> You can set a custom domain for the Netlify landing page and point `/app` to your Streamlit URL via the redirect in `netlify.toml`.

---

## Why Two Deployments?

Streamlit is a Python web server — it cannot run inside Netlify's serverless/static infrastructure. The architecture used here is:

```
Netlify (static)          Streamlit Cloud (Python)
  Landing page      →         ML Dashboard
  index.html              app.py  +  loan_approval_data.csv
```

This is the standard approach for ML projects wanting a polished public-facing entry point on Netlify.

---

## Features

- **Live prediction** — enter any applicant's details, get instant Approved / Rejected verdict with confidence %
- **Probability donut chart** — visualise the approval vs rejection probability split
- **Credit Score & DTI gauges** — interactive Plotly gauge meters
- **Model comparison bar chart** — Accuracy, Precision, Recall, F1 across all three models
- **Confusion matrices** — interactive heatmaps for each classifier
- **Per-model accuracy gauges** — side-by-side visual comparison
- **Dark premium UI** — Cormorant Garamond + Outfit fonts, gold accent system, glass-morphism cards
- **Responsive** — works on desktop and tablet

---

## Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

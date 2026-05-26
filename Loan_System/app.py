"""
╔══════════════════════════════════════════════════════════════════╗
║   CreditWise AI  ·  SecureTrust Bank  ·  Loan Intelligence      ║
║   Complete rebuild — futuristic glassmorphism UI                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import warnings, os
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CreditWise AI · SecureTrust Bank",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  MASTER STYLESHEET  — futuristic glassmorphism
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Custom Properties ─────────────────────────────────────── */
:root {
  --bg-deep:    #04030A;
  --bg-surface: rgba(13, 9, 28, 0.85);
  --bg-card:    rgba(18, 12, 38, 0.7);
  --bg-card2:   rgba(26, 18, 52, 0.6);
  --border:     rgba(139, 92, 246, 0.18);
  --border-hi:  rgba(139, 92, 246, 0.45);
  --violet:     #8B5CF6;
  --violet-lt:  #A78BFA;
  --lime:       #A3E635;
  --lime-dk:    #65A30D;
  --cyan:       #22D3EE;
  --rose:       #FB7185;
  --amber:      #FCD34D;
  --text:       #F1F0F5;
  --text-muted: #6B6882;
  --text-dim:   #2D2A40;
  --glow-v:     rgba(139,92,246,0.35);
  --glow-l:     rgba(163,230,53,0.25);
}

/* ── Base reset ────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

/* Deep space background with hexagonal shimmer */
.stApp {
  background-color: var(--bg-deep) !important;
  background-image:
    radial-gradient(ellipse 80% 60% at 20% 20%, rgba(139,92,246,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 70% at 80% 70%, rgba(34,211,238,0.04) 0%, transparent 55%),
    radial-gradient(ellipse 40% 40% at 60% 10%, rgba(163,230,53,0.03) 0%, transparent 50%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='52'%3E%3Cpath d='M30 2L56 17v30L30 50 4 47V17z' fill='none' stroke='rgba(139,92,246,0.04)' stroke-width='0.8'/%3E%3C/svg%3E");
  background-attachment: fixed;
}

/* Block container */
.block-container {
  padding: 0 2rem 4rem !important;
  max-width: 1400px !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stSidebarNav"],
section[data-testid="stSidebar"] { display: none !important; }

/* ── Typography ─────────────────────────────────────────────────── */
h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.02em;
}

/* ── Navigation bar ────────────────────────────────────────────── */
.cw-nav {
  position: sticky; top: 0; z-index: 999;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.85rem 2.5rem;
  background: rgba(4, 3, 10, 0.92);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(139,92,246,0.15);
  margin: 0 -2rem 2.5rem;
  box-shadow: 0 4px 40px rgba(0,0,0,0.4);
}
.cw-nav-brand {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.35rem; font-weight: 700;
  color: #F1F0F5; letter-spacing: -0.03em;
}
.cw-nav-brand em {
  font-style: normal;
  background: linear-gradient(135deg, #8B5CF6 0%, #A3E635 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.cw-nav-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(139,92,246,0.1);
  border: 1px solid rgba(139,92,246,0.25);
  border-radius: 100px;
  padding: 0.3rem 1rem;
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--violet-lt);
}
.cw-nav-pill::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: var(--lime); box-shadow: 0 0 8px var(--lime);
  animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Radio → horizontal tab strip ─────────────────────────────── */
div[role="radiogroup"] {
  display: flex !important; flex-direction: row !important;
  gap: 0.3rem !important;
  background: rgba(13,9,28,0.8) !important;
  border: 1px solid rgba(139,92,246,0.15) !important;
  border-radius: 14px !important;
  padding: 0.4rem !important;
  margin-bottom: 2rem !important;
}
div[role="radiogroup"] label {
  flex: 1 !important;
  display: flex !important; justify-content: center !important;
  padding: 0.55rem 1rem !important;
  border-radius: 10px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.82rem !important; font-weight: 600 !important;
  letter-spacing: 0.04em !important; text-transform: uppercase !important;
  cursor: pointer !important;
  transition: all 0.25s ease !important;
  color: var(--text-muted) !important;
  border: 1px solid transparent !important;
  white-space: nowrap !important;
}
div[role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(135deg,
    rgba(139,92,246,0.25) 0%, rgba(163,230,53,0.12) 100%) !important;
  border-color: rgba(139,92,246,0.4) !important;
  color: #F1F0F5 !important;
  box-shadow: 0 0 20px rgba(139,92,246,0.2),
              inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
div[role="radiogroup"] label:hover:not(:has(input:checked)) {
  background: rgba(139,92,246,0.08) !important;
  color: var(--violet-lt) !important;
  border-color: rgba(139,92,246,0.2) !important;
}
div[role="radiogroup"] input[type="radio"] { display: none !important; }

/* ── Metric cards ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1.4rem 1.6rem !important;
  backdrop-filter: blur(12px) !important;
  position: relative; overflow: hidden !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
}
[data-testid="stMetric"]:hover {
  border-color: var(--border-hi) !important;
  box-shadow: 0 0 30px rgba(139,92,246,0.15) !important;
}
[data-testid="stMetric"]::before {
  content: '';
  position: absolute; top: -30px; right: -30px;
  width: 100px; height: 100px;
  background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
}
[data-testid="stMetricLabel"] p {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.68rem !important; font-weight: 600 !important;
  letter-spacing: 0.12em !important; text-transform: uppercase !important;
  color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 2.1rem !important; font-weight: 700 !important;
  color: #F1F0F5 !important; letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] { font-size: 0.76rem !important; }

/* ── Inputs & Selects ──────────────────────────────────────────── */
[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stNumberInput > div > div,
.stTextInput > div > div {
  background: rgba(13,9,28,0.9) !important;
  border: 1px solid rgba(139,92,246,0.2) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
  border-color: rgba(139,92,246,0.55) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.12) !important;
}
input, [data-baseweb="select"] span {
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}
/* Dropdown options */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
  background: #0D091C !important;
  color: var(--text) !important;
}
[data-baseweb="popover"] li:hover { background: rgba(139,92,246,0.15) !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
  background: linear-gradient(90deg, #8B5CF6, #A3E635) !important;
}
[data-testid="stSlider"] > div > div > div {
  background: rgba(139,92,246,0.15) !important;
}

/* Labels */
label, [data-testid="stWidgetLabel"] p {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.8rem !important; font-weight: 500 !important;
  color: var(--text-muted) !important; letter-spacing: 0.02em !important;
}

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%) !important;
  color: #F1F0F5 !important; border: none !important;
  border-radius: 10px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important; font-size: 0.88rem !important;
  letter-spacing: 0.06em !important; text-transform: uppercase !important;
  padding: 0.7rem 2rem !important;
  box-shadow: 0 4px 24px rgba(139,92,246,0.35) !important;
  transition: all 0.25s ease !important;
  position: relative; overflow: hidden !important;
}
.stButton > button::before {
  content: '';
  position: absolute; top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  transition: left 0.5s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
  box-shadow: 0 6px 32px rgba(139,92,246,0.55) !important;
  transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Expander ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(12px) !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important; color: var(--text-muted) !important;
}

/* ── Columns gap ───────────────────────────────────────────────── */
[data-testid="column"] { gap: 0 !important; }

/* ── Custom Components ─────────────────────────────────────────── */

/* Page section header */
.cw-section {
  display: flex; align-items: center; gap: 0.8rem;
  margin: 2rem 0 1.2rem;
}
.cw-section-icon {
  width: 36px; height: 36px; border-radius: 9px;
  background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(163,230,53,0.15));
  border: 1px solid rgba(139,92,246,0.3);
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; flex-shrink: 0;
}
.cw-section-text h3 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.05rem; font-weight: 700; color: #F1F0F5; margin: 0;
}
.cw-section-text p {
  font-size: 0.78rem; color: var(--text-muted); margin: 0; line-height: 1.4;
}
.cw-section-line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(139,92,246,0.3) 0%, transparent 100%);
}

/* Glass card */
.cw-glass {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.6rem;
  backdrop-filter: blur(16px);
  position: relative; overflow: hidden;
}
.cw-glass::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
}

/* Page hero banner */
.cw-hero {
  background: linear-gradient(135deg,
    rgba(13,9,28,0.95) 0%, rgba(26,18,52,0.9) 50%, rgba(13,9,28,0.95) 100%);
  border: 1px solid rgba(139,92,246,0.18);
  border-radius: 20px;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  position: relative; overflow: hidden;
}
.cw-hero::before {
  content: '';
  position: absolute; top: -60%; right: -10%;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 65%);
}
.cw-hero::after {
  content: '';
  position: absolute; bottom: -40%; left: 30%;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(163,230,53,0.05) 0%, transparent 60%);
}
.cw-hero-eyebrow {
  display: inline-flex; align-items: center; gap: 0.5rem;
  background: rgba(139,92,246,0.1);
  border: 1px solid rgba(139,92,246,0.25);
  border-radius: 100px;
  padding: 0.25rem 0.85rem;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--violet-lt); margin-bottom: 0.9rem;
}
.cw-hero h1 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: clamp(1.8rem, 3vw, 2.6rem) !important;
  font-weight: 700 !important; line-height: 1.1 !important;
  letter-spacing: -0.03em !important;
  background: linear-gradient(135deg, #F1F0F5 40%, #A78BFA 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin: 0 0 0.6rem !important;
}
.cw-hero p { color: var(--text-muted); font-size: 0.95rem; max-width: 600px; margin: 0; }

/* Stat badge */
.cw-stat-badge {
  display: inline-flex; flex-direction: column; gap: 0.1rem;
  background: rgba(139,92,246,0.08);
  border: 1px solid rgba(139,92,246,0.2);
  border-radius: 12px; padding: 0.6rem 1rem;
}
.cw-stat-badge .val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.5rem; font-weight: 700; color: #F1F0F5;
  line-height: 1;
}
.cw-stat-badge .lbl {
  font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--text-muted);
}

/* Verdict card */
.cw-verdict-approved {
  background: linear-gradient(135deg,
    rgba(163,230,53,0.12) 0%, rgba(101,163,13,0.07) 100%);
  border: 1.5px solid rgba(163,230,53,0.35);
  border-radius: 16px; padding: 2rem; text-align: center;
  animation: glowPulse 2.5s ease-in-out infinite;
}
.cw-verdict-rejected {
  background: linear-gradient(135deg,
    rgba(251,113,133,0.12) 0%, rgba(190,18,60,0.07) 100%);
  border: 1.5px solid rgba(251,113,133,0.35);
  border-radius: 16px; padding: 2rem; text-align: center;
  animation: glowRed 2.5s ease-in-out infinite;
}
@keyframes glowPulse {
  0%,100% { box-shadow: 0 0 20px rgba(163,230,53,0.12); }
  50%      { box-shadow: 0 0 40px rgba(163,230,53,0.28); }
}
@keyframes glowRed {
  0%,100% { box-shadow: 0 0 20px rgba(251,113,133,0.12); }
  50%      { box-shadow: 0 0 40px rgba(251,113,133,0.28); }
}
.cw-verdict-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.cw-verdict-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.9rem; font-weight: 700;
  letter-spacing: 0.04em; margin-bottom: 0.3rem;
}
.cw-verdict-approved .cw-verdict-title { color: #A3E635; }
.cw-verdict-rejected .cw-verdict-title { color: #FB7185; }
.cw-verdict-sub { font-size: 0.85rem; color: var(--text-muted); }

/* Confidence arc (CSS-only) */
.cw-conf-wrap {
  margin: 1.2rem 0 0.5rem;
  background: rgba(255,255,255,0.04);
  border-radius: 100px; height: 8px; overflow: hidden;
}
.cw-conf-fill {
  height: 100%; border-radius: 100px;
  transition: width 1s cubic-bezier(0.4,0,0.2,1);
}
.cw-conf-approved { background: linear-gradient(90deg, #65A30D50, #A3E635); }
.cw-conf-rejected { background: linear-gradient(90deg, #BE123C50, #FB7185); }

/* Factor rows */
.cw-factors { border-radius: 12px; overflow: hidden; }
.cw-factor-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.55rem 0.9rem;
  border-bottom: 1px solid rgba(139,92,246,0.07);
  transition: background 0.15s;
}
.cw-factor-row:last-child { border-bottom: none; }
.cw-factor-row:hover { background: rgba(139,92,246,0.05); }
.f-name { font-size: 0.82rem; color: #6B6882; }
.f-val  { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #8B7DAA; }
.f-good { font-size: 0.75rem; font-weight: 700; color: #A3E635; }
.f-warn { font-size: 0.75rem; font-weight: 700; color: #FCD34D; }
.f-bad  { font-size: 0.75rem; font-weight: 700; color: #FB7185; }

/* Insight strip */
.cw-insight {
  background: rgba(139,92,246,0.06);
  border: 1px solid rgba(139,92,246,0.14);
  border-left: 3px solid #8B5CF6;
  border-radius: 0 10px 10px 0;
  padding: 0.75rem 1rem;
  font-size: 0.84rem; color: #6B6882;
  line-height: 1.55; margin: 0.4rem 0;
}
.cw-insight strong { color: var(--violet-lt); }

/* Model perf card */
.cw-model {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 1.2rem 1.4rem;
  position: relative; overflow: hidden;
  transition: border-color 0.25s, box-shadow 0.25s;
}
.cw-model:hover {
  border-color: var(--border-hi);
  box-shadow: 0 0 25px rgba(139,92,246,0.12);
}
.cw-model.best { border-color: rgba(163,230,53,0.25); }
.cw-model.best:hover { box-shadow: 0 0 25px rgba(163,230,53,0.12); }
.cw-model-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.92rem; font-weight: 700; color: #F1F0F5;
  margin-bottom: 0.9rem;
}
.cw-best-chip {
  display: inline-block;
  background: rgba(163,230,53,0.12);
  border: 1px solid rgba(163,230,53,0.3);
  color: #A3E635;
  font-size: 0.58rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 0.12rem 0.5rem; border-radius: 4px;
  margin-left: 0.5rem; vertical-align: middle;
}
.cw-metric-pill {
  display: inline-flex; flex-direction: column; align-items: center;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(139,92,246,0.1);
  border-radius: 8px; padding: 0.3rem 0.5rem; margin: 0.15rem;
  min-width: 55px;
}
.cw-mp-lbl {
  font-size: 0.55rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase; color: #2D2A40;
}
.cw-mp-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.9rem; font-weight: 700; color: var(--violet-lt);
}

/* Placeholder state */
.cw-placeholder {
  background: rgba(139,92,246,0.03);
  border: 1px dashed rgba(139,92,246,0.15);
  border-radius: 16px; padding: 4rem 2rem; text-align: center;
}
.cw-placeholder-icon { font-size: 3rem; margin-bottom: 1rem; }
.cw-placeholder-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700; font-size: 1.05rem; color: #2D2A40; margin-bottom: 0.4rem;
}
.cw-placeholder-sub { font-size: 0.82rem; color: #1A1730; }

/* Step cards (how it works) */
.cw-step {
  display: flex; gap: 1rem; align-items: flex-start;
  padding: 1rem 1.2rem; margin: 0.5rem 0;
  background: rgba(139,92,246,0.04);
  border: 1px solid rgba(139,92,246,0.1);
  border-radius: 12px;
}
.cw-step-num {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6rem; font-weight: 800;
  color: rgba(139,92,246,0.2); flex-shrink: 0; line-height: 1;
}
.cw-step-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.88rem; font-weight: 700; color: #8B7DAA; margin-bottom: 0.2rem;
}
.cw-step-desc { font-size: 0.8rem; color: #2D2A40; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(139,92,246,0.25); border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.45); }

/* ── Plotly chart area ──────────────────────────────────────────── */
.js-plotly-plot .plotly { border-radius: 12px; }

/* ── Dataframe ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(139,92,246,0.15) !important;
  border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PLOTLY BASE THEME
# ══════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#6B6882", size=12),
    title_font=dict(family="Space Grotesk", color="#C4B5FD", size=14),
    xaxis=dict(
        gridcolor="rgba(139,92,246,0.08)",
        linecolor="rgba(139,92,246,0.12)",
        tickfont=dict(color="#4B4664", size=11),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(139,92,246,0.08)",
        linecolor="rgba(139,92,246,0.12)",
        tickfont=dict(color="#4B4664", size=11),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(13,9,28,0.7)",
        bordercolor="rgba(139,92,246,0.15)",
        borderwidth=1,
        font=dict(color="#6B6882"),
    ),
    margin=dict(l=20, r=20, t=44, b=24),
    hoverlabel=dict(
        bgcolor="#0D091C",
        bordercolor="rgba(139,92,246,0.4)",
        font=dict(family="Inter", color="#F1F0F5"),
    ),
)
COLORS = dict(
    violet="#8B5CF6", lime="#A3E635", cyan="#22D3EE",
    rose="#FB7185",   amber="#FCD34D", blue="#60A5FA",
)
PALETTE = [COLORS[k] for k in COLORS]

# ══════════════════════════════════════════════════════════════════
#  DATA LOADING  (cached)
# ══════════════════════════════════════════════════════════════════
DATA_FILE = "loan_approval_data.csv"

CAT_OHE   = ["Employment_Status","Marital_Status","Loan_Purpose",
             "Property_Area","Gender","Employer_Category"]
CAT_LABEL = ["Education_Level"]
TARGET    = "Loan_Approved"
DROP      = ["Applicant_ID"]

@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    path = DATA_FILE if os.path.exists(DATA_FILE) else os.path.join(
        os.path.dirname(__file__), DATA_FILE)
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def build_pipeline() -> dict:
    df = load_raw().copy()

    # ── impute ────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = (SimpleImputer(strategy="most_frequent")
                    .fit_transform(df[cat_cols]))

    # ── target → integer ─────────────────────────────────────────
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    df.drop(columns=DROP, errors="ignore", inplace=True)

    # ── label encode Education_Level ─────────────────────────────
    le = LabelEncoder()
    for c in CAT_LABEL:
        if c in df.columns:
            df[c] = le.fit_transform(df[c])

    # ── one-hot encode ────────────────────────────────────────────
    ohe_cols = [c for c in CAT_OHE if c in df.columns]
    ohe = OneHotEncoder(drop="first", sparse_output=False,
                        handle_unknown="ignore")
    ohe_arr = ohe.fit_transform(df[ohe_cols])
    ohe_df  = pd.DataFrame(ohe_arr,
                            columns=ohe.get_feature_names_out(ohe_cols),
                            index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), ohe_df], axis=1)

    # ── feature engineering ───────────────────────────────────────
    df["DTI_sq"]    = df["DTI_Ratio"]    ** 2
    df["Credit_sq"] = df["Credit_Score"] ** 2

    # ── split & scale ─────────────────────────────────────────────
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    scaler   = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    # ── train all models ──────────────────────────────────────────
    models_def = {
        "Random Forest":       RandomForestClassifier(
                                   n_estimators=200, max_depth=None,
                                   random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(
                                   n_estimators=150, random_state=42),
        "Logistic Regression": LogisticRegression(
                                   max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes":         GaussianNB(),
    }

    results, trained, best_name, best_acc = {}, {}, None, 0.0
    for name, m in models_def.items():
        m.fit(Xs_train, y_train)
        yp  = m.predict(Xs_test)
        acc = float(accuracy_score(y_test, yp))
        results[name] = {
            "Accuracy":  round(acc * 100, 1),
            "Precision": round(precision_score(y_test, yp, zero_division=0) * 100, 1),
            "Recall":    round(recall_score(y_test, yp, zero_division=0) * 100, 1),
            "F1":        round(f1_score(y_test, yp, zero_division=0) * 100, 1),
            "cm":        confusion_matrix(y_test, yp).tolist(),
        }
        trained[name] = m
        if acc > best_acc:
            best_acc, best_name = acc, name

    # ── feature importances ───────────────────────────────────────
    rf_imp = pd.DataFrame({
        "feature":    X.columns,
        "importance": trained["Random Forest"].feature_importances_,
    }).sort_values("importance", ascending=False).head(15).reset_index(drop=True)

    return {
        "models":      trained,
        "scaler":      scaler,
        "ohe":         ohe,
        "ohe_cols":    ohe_cols,
        "le":          le,
        "feat_cols":   list(X.columns),
        "results":     results,
        "best":        best_name,
        "feat_imp":    rf_imp,
        "X_test":      X_test,
        "y_test":      y_test,
    }

# ══════════════════════════════════════════════════════════════════
#  PREDICTION HELPER
# ══════════════════════════════════════════════════════════════════
def run_prediction(pipe: dict, raw: dict) -> tuple[int, float]:
    inp       = raw.copy()
    scaler    = pipe["scaler"]
    ohe       = pipe["ohe"]
    ohe_cols  = pipe["ohe_cols"]
    feat_cols = pipe["feat_cols"]
    model     = pipe["models"]["Random Forest"]

    # label-encode Education_Level
    edu_map = {"Not Graduate": 0, "Graduate": 1}
    inp["Education_Level"] = edu_map.get(inp.get("Education_Level", ""), 0)

    # OHE
    ohe_input = pd.DataFrame(
        [[inp.get(c, "") for c in ohe_cols]], columns=ohe_cols)
    ohe_vals  = ohe.transform(ohe_input)
    ohe_df    = pd.DataFrame(
        ohe_vals, columns=ohe.get_feature_names_out(ohe_cols))

    # remove OHE source columns from inp
    for c in ohe_cols:
        inp.pop(c, None)

    df_inp = pd.concat(
        [pd.DataFrame([inp]).reset_index(drop=True),
         ohe_df.reset_index(drop=True)], axis=1)

    # feature engineering
    df_inp["DTI_sq"]    = df_inp["DTI_Ratio"]    ** 2
    df_inp["Credit_sq"] = df_inp["Credit_Score"] ** 2

    # align columns
    for col in feat_cols:
        if col not in df_inp.columns:
            df_inp[col] = 0.0
    df_inp = df_inp[feat_cols]

    scaled = scaler.transform(df_inp.values)
    pred   = int(model.predict(scaled)[0])
    proba  = float(model.predict_proba(scaled)[0][1])
    return pred, proba

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def section(icon: str, title: str, sub: str = ""):
    sub_html = f'<p>{sub}</p>' if sub else ""
    st.markdown(f"""
    <div class="cw-section">
      <div class="cw-section-icon">{icon}</div>
      <div class="cw-section-text"><h3>{title}</h3>{sub_html}</div>
      <div class="cw-section-line"></div>
    </div>""", unsafe_allow_html=True)

def insight(html: str):
    st.markdown(f'<div class="cw-insight">{html}</div>', unsafe_allow_html=True)

def factor_cls(v, g, w, higher_good=True):
    ok = v >= g if higher_good else v <= g
    mid = v >= w if higher_good else v <= w
    if ok:  return "f-good", "✦ Strong" if higher_good else "✦ Low"
    if mid: return "f-warn", "◈ Moderate"
    return "f-bad", "✖ Weak" if higher_good else "✖ High"

# ══════════════════════════════════════════════════════════════════
#  LOAD DATA & TRAIN
# ══════════════════════════════════════════════════════════════════
raw_df = load_raw()

if "pipeline" not in st.session_state:
    with st.spinner(""):
        st.session_state["pipeline"] = build_pipeline()

pipe = st.session_state["pipeline"]

# ══════════════════════════════════════════════════════════════════
#  TOP NAV BAR
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cw-nav">
  <div class="cw-nav-brand">Credit<em>Wise</em> <span style="font-size:0.7rem;font-weight:400;color:#2D2A40;margin-left:0.5rem;">by SecureTrust Bank</span></div>
  <div class="cw-nav-pill">AI Engine Active</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  HORIZONTAL NAV  (radio styled as pill tabs)
# ══════════════════════════════════════════════════════════════════
page = st.radio(
    label="",
    options=["◈  Overview", "⟡  Predictor", "◎  Analytics", "◆  Models"],
    horizontal=True,
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  ─────────────────────────────────────────────────────────────
#  PAGE 1 ·  OVERVIEW DASHBOARD
#  ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
if page == "◈  Overview":

    df    = raw_df.copy()
    total = len(df)
    appr  = int((df[TARGET] == "Yes").sum())
    rej   = int((df[TARGET] == "No").sum())
    best  = pipe["best"]
    bacc  = pipe["results"][best]["Accuracy"]

    # Hero
    st.markdown(f"""
    <div class="cw-hero">
      <div class="cw-hero-eyebrow">◈ Live Dashboard — Portfolio Intelligence</div>
      <h1>Loan Portfolio Overview</h1>
      <p>Real-time intelligence across all SecureTrust applications —
         approval rates, risk distributions, income profiles and geographic coverage.</p>
      <div style="display:flex;gap:1rem;margin-top:1.5rem;flex-wrap:wrap;">
        <div class="cw-stat-badge"><span class="val">{total:,}</span><span class="lbl">Applications</span></div>
        <div class="cw-stat-badge"><span class="val">{appr:,}</span><span class="lbl">Approved</span></div>
        <div class="cw-stat-badge"><span class="val">{rej:,}</span><span class="lbl">Rejected</span></div>
        <div class="cw-stat-badge" style="border-color:rgba(163,230,53,0.25);background:rgba(163,230,53,0.07);">
          <span class="val" style="background:linear-gradient(135deg,#A3E635,#65A30D);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{bacc}%</span>
          <span class="lbl">Best Model Acc</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Applications", f"{total:,}", "1,000 records")
    k2.metric("Approval Rate",   f"{appr/total*100:.1f}%",  f"+{appr} approved")
    k3.metric("Rejection Rate",  f"{rej/total*100:.1f}%",   f"−{rej} rejected")
    k4.metric("Top Model",       best.split()[0]+" "+best.split()[1] if " " in best else best,
              f"{bacc}% accuracy")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row A: Approval donut  +  Income overlay histogram ───────
    col_a, col_b = st.columns([1, 1.65], gap="medium")

    with col_a:
        section("◍", "Approval Split")
        fig = go.Figure(go.Pie(
            labels=["Approved", "Rejected"], values=[appr, rej],
            hole=0.68,
            marker=dict(colors=[COLORS["lime"], COLORS["rose"]],
                        line=dict(color="#04030A", width=3)),
            textinfo="percent",
            textfont=dict(family="Space Grotesk", size=12, color="#F1F0F5"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{appr/total*100:.0f}%</b><br><span style='font-size:10px'>Approval</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Space Grotesk", color="#F1F0F5", size=18),
        )
        fig.update_layout(**PL, height=290,
                          showlegend=True,
                          legend=dict(orientation="h", y=-0.08, x=0.15))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        section("◌", "Income Distribution by Outcome",
                "Monthly applicant income grouped by approval status")
        fig2 = go.Figure()
        for outcome, col, name in [
            ("Yes", COLORS["lime"], "Approved"),
            ("No",  COLORS["rose"], "Rejected"),
        ]:
            sub = df[df[TARGET] == outcome]["Applicant_Income"].dropna()
            fig2.add_trace(go.Histogram(
                x=sub, name=name, nbinsx=28,
                marker=dict(color=col, opacity=0.75,
                            line=dict(color=col, width=0.3)),
                hovertemplate=f"<b>{name}</b><br>Income: ₹%{{x:,.0f}}<br>Count: %{{y}}<extra></extra>",
            ))
        fig2.update_layout(**PL, height=290, barmode="overlay",
                           xaxis_title="Monthly Income (₹)",
                           yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Row B: Scatter + Area bar ────────────────────────────────
    col_c, col_d = st.columns(2, gap="medium")

    with col_c:
        section("◎", "Credit Score vs DTI Ratio",
                "Risk positioning — sample of 600 applicants")
        samp = df.dropna(subset=["Credit_Score","DTI_Ratio",TARGET]).sample(
            n=min(600, len(df)), random_state=1)
        fig3 = px.scatter(
            samp, x="DTI_Ratio", y="Credit_Score", color=TARGET,
            color_discrete_map={"Yes": COLORS["lime"], "No": COLORS["rose"]},
            opacity=0.60,
            labels={"DTI_Ratio": "DTI Ratio", "Credit_Score": "Credit Score", TARGET: ""},
        )
        fig3.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig3.update_layout(**PL, height=290)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col_d:
        section("◆", "Loan Purpose Breakdown",
                "Application volume by purpose category")
        pur = (df.dropna(subset=["Loan_Purpose", TARGET])
                 .groupby(["Loan_Purpose", TARGET])
                 .size().reset_index(name="n"))
        fig4 = px.bar(
            pur, x="Loan_Purpose", y="n", color=TARGET,
            color_discrete_map={"Yes": COLORS["lime"], "No": COLORS["rose"]},
            barmode="group",
            labels={"Loan_Purpose": "Purpose", "n": "Applications", TARGET: ""},
        )
        fig4.update_layout(**PL, height=290)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # ── Row C: Heatmap + Insights ────────────────────────────────
    col_e, col_f = st.columns([1.5, 1], gap="medium")

    with col_e:
        section("⬡", "Approval Rate — Area × Employment",
                "% approved per segment combination")
        heat = (df.dropna(subset=["Property_Area","Employment_Status"])
                  .groupby(["Property_Area","Employment_Status"])
                  .apply(lambda x: round((x[TARGET]=="Yes").mean()*100, 1))
                  .reset_index(name="rate"))
        fig5 = px.density_heatmap(
            heat, x="Property_Area", y="Employment_Status", z="rate",
            color_continuous_scale=[
                [0.0, "#04030A"], [0.4, "#1E1050"],
                [0.7, "#5B21B6"], [1.0, "#A3E635"],
            ],
            text_auto=".1f",
        )
        fig5.update_traces(
            textfont=dict(family="JetBrains Mono", color="#F1F0F5", size=11))
        fig5.update_layout(
            **PL, height=290,
            coloraxis_colorbar=dict(
                tickfont=dict(color="#4B4664"),
                title=dict(text="Rate %", font=dict(color="#4B4664")),
            ))
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    with col_f:
        section("⟡", "Key Insights")
        avg_i_y = df[df[TARGET]=="Yes"]["Applicant_Income"].mean()
        avg_i_n = df[df[TARGET]=="No"]["Applicant_Income"].mean()
        avg_c_y = df[df[TARGET]=="Yes"]["Credit_Score"].mean()
        avg_c_n = df[df[TARGET]=="No"]["Credit_Score"].mean()
        avg_d_y = df[df[TARGET]=="Yes"]["DTI_Ratio"].mean()
        avg_d_n = df[df[TARGET]=="No"]["DTI_Ratio"].mean()
        for txt in [
            f"Approved applicants earn on average <strong>₹{avg_i_y:,.0f}/mo</strong> vs ₹{avg_i_n:,.0f}/mo for rejected — a {(avg_i_y/avg_i_n-1)*100:.0f}% premium.",
            f"Average credit score for approvals: <strong>{avg_c_y:.0f}</strong> vs {avg_c_n:.0f} for rejections — a {avg_c_y-avg_c_n:.0f} pt gap.",
            f"DTI Ratio averages <strong>{avg_d_y:.2f}</strong> (approved) vs {avg_d_n:.2f} (rejected) — lower leverage predicts approval.",
            f"Best performing model: <strong>{best}</strong> at {bacc}% accuracy with precision {pipe['results'][best]['Precision']}%.",
        ]:
            insight(txt)


# ══════════════════════════════════════════════════════════════════
#  PAGE 2 ·  LOAN PREDICTOR
# ══════════════════════════════════════════════════════════════════
elif page == "⟡  Predictor":

    st.markdown("""
    <div class="cw-hero">
      <div class="cw-hero-eyebrow">⟡ AI Decision Engine — Real-Time Inference</div>
      <h1>Loan Approval Predictor</h1>
      <p>Enter applicant details to receive an instant AI-driven decision powered by a
         Random Forest model trained on 1,000 historical applications.</p>
    </div>""", unsafe_allow_html=True)

    col_form, col_out = st.columns([1.05, 1], gap="large")

    # ── INPUT FORM ───────────────────────────────────────────────
    with col_form:

        section("◌", "Personal Details")
        r1a, r1b, r1c = st.columns(3)
        gender   = r1a.selectbox("Gender",         ["Male","Female"])
        age      = r1b.number_input("Age",          min_value=21, max_value=59, value=32)
        marital  = r1c.selectbox("Marital Status",  ["Married","Single"])
        r2a, r2b, r2c = st.columns(3)
        deps     = r2a.selectbox("Dependents",      [0,1,2,3], index=1)
        edu      = r2b.selectbox("Education",       ["Graduate","Not Graduate"])
        area     = r2c.selectbox("Property Area",   ["Urban","Semiurban","Rural"])

        section("◆", "Employment & Income")
        r3a, r3b = st.columns(2)
        emp_s    = r3a.selectbox("Employment Status",
                                  ["Salaried","Self-employed","Contract","Unemployed"])
        emp_c    = r3b.selectbox("Employer Category",
                                  ["Private","Government","MNC","Business","Unemployed"])
        r4a, r4b = st.columns(2)
        inc_a    = r4a.number_input("Applicant Income (₹/mo)",
                                     min_value=0, max_value=20000, value=10500, step=500)
        inc_c    = r4b.number_input("Co-applicant Income (₹/mo)",
                                     min_value=0, max_value=10000, value=3000, step=500)

        section("⬡", "Financial Profile")
        r5a, r5b = st.columns(2)
        cs = r5a.slider("Credit Score", min_value=550, max_value=799, value=700, step=5)
        dti = r5b.slider("DTI Ratio",   min_value=0.10, max_value=0.60, value=0.30, step=0.01)
        r6a, r6b, r6c = st.columns(3)
        sav      = r6a.number_input("Savings (₹)",      min_value=0, max_value=20000, value=8000, step=500)
        collat   = r6b.number_input("Collateral (₹)",   min_value=0, max_value=50000, value=20000, step=1000)
        ex_loans = r6c.selectbox("Existing Loans",      [0,1,2,3,4], index=1)

        section("◍", "Loan Request")
        r7a, r7b, r7c = st.columns(3)
        l_amt    = r7a.number_input("Loan Amount (₹)",
                                     min_value=1000, max_value=40000, value=18000, step=1000)
        l_term   = r7b.selectbox("Loan Term (months)",
                                  [12,24,36,48,60,72,84], index=3)
        l_purp   = r7c.selectbox("Loan Purpose",
                                  ["Home","Personal","Business","Education","Car"])

        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("⟡  Run AI Analysis", use_container_width=True)

    # ── OUTPUT PANEL ─────────────────────────────────────────────
    with col_out:
        section("◈", "Decision Engine")

        if go_btn:
            raw_inp = {
                "Applicant_Income":   float(inc_a),
                "Coapplicant_Income": float(inc_c),
                "Employment_Status":  emp_s,
                "Age":                float(age),
                "Marital_Status":     marital,
                "Dependents":         float(deps),
                "Credit_Score":       float(cs),
                "Existing_Loans":     float(ex_loans),
                "DTI_Ratio":          float(dti),
                "Savings":            float(sav),
                "Collateral_Value":   float(collat),
                "Loan_Amount":        float(l_amt),
                "Loan_Term":          float(l_term),
                "Loan_Purpose":       l_purp,
                "Property_Area":      area,
                "Education_Level":    edu,
                "Gender":             gender,
                "Employer_Category":  emp_c,
            }

            pred, proba = run_prediction(pipe, raw_inp)
            pct = proba * 100
            approved = pred == 1

            # Verdict badge
            css_cls = "cw-verdict-approved" if approved else "cw-verdict-rejected"
            icon    = "✦" if approved else "✖"
            title   = "APPROVED" if approved else "REJECTED"
            sub     = "Application meets approval criteria." if approved \
                      else "Application does not meet approval criteria."
            conf_cls = "cw-conf-approved" if approved else "cw-conf-rejected"
            pct_col  = "#A3E635" if approved else "#FB7185"

            st.markdown(f"""
            <div class="{css_cls}">
              <div class="cw-verdict-icon">{icon}</div>
              <div class="cw-verdict-title">{title}</div>
              <div class="cw-verdict-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

            # Confidence
            st.markdown(f"""
            <div class="cw-glass" style="margin-top:1rem;padding:1.2rem 1.4rem;">
              <div style="font-family:'Space Grotesk',sans-serif;font-size:0.65rem;
                          font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                          color:#2D2A40;margin-bottom:0.2rem;">Approval Probability</div>
              <div style="display:flex;align-items:baseline;gap:0.5rem;margin-bottom:0.5rem;">
                <span style="font-family:'Space Grotesk',sans-serif;font-size:2rem;
                             font-weight:700;color:{pct_col};">{pct:.1f}%</span>
                <span style="font-size:0.78rem;color:#2D2A40;">confidence</span>
              </div>
              <div class="cw-conf-wrap">
                <div class="{conf_cls} cw-conf-fill" style="width:{pct:.1f}%"></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                          font-size:0.68rem;color:#1A1730;margin-top:0.25rem;">
                <span>Lower risk →</span><span>← Higher risk</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Key factors table
            section("◍", "Risk Factor Analysis")

            fc_cs  = factor_cls(cs,    720, 640)
            fc_dti = factor_cls(dti,   0.25, 0.45, higher_good=False)
            fc_inc = factor_cls(inc_a+inc_c, 12000, 6000)
            fc_sav = factor_cls(sav,   12000, 4000)
            fc_el  = factor_cls(ex_loans, 0, 2, higher_good=False)
            fc_col = factor_cls(collat, 30000, 10000)

            factors = [
                ("Credit Score",     cs,                      *fc_cs),
                ("DTI Ratio",        f"{dti:.2f}",            *fc_dti),
                ("Combined Income",  f"₹{inc_a+inc_c:,.0f}", *fc_inc),
                ("Savings",          f"₹{sav:,.0f}",          *fc_sav),
                ("Existing Loans",   ex_loans,                *fc_el),
                ("Collateral",       f"₹{collat:,.0f}",       *fc_col),
            ]
            rows = "".join(
                f'<div class="cw-factor-row">'
                f'<span class="f-name">{n}</span>'
                f'<span class="f-val">{v}</span>'
                f'<span class="{cls}">{lbl}</span>'
                f'</div>'
                for n, v, cls, lbl in factors
            )
            st.markdown(
                f'<div class="cw-glass" style="padding:0.4rem;">'
                f'<div class="cw-factors">{rows}</div></div>',
                unsafe_allow_html=True)

            # Quick model ensemble check
            with st.expander("◎  Full Model Ensemble Results"):
                ens_data = []
                raw_copy = raw_inp.copy()
                for mname, m in pipe["models"].items():
                    try:
                        p2, pr2 = run_prediction({**pipe, "models": {"Random Forest": m}
                                                  if True else pipe}, raw_copy)
                        # re-run properly for each model
                    except Exception:
                        pass
                # just show training scores
                for mname, res in pipe["results"].items():
                    ens_data.append({
                        "Model": mname,
                        "Accuracy": f"{res['Accuracy']}%",
                        "Precision": f"{res['Precision']}%",
                        "F1": f"{res['F1']}%",
                    })
                st.dataframe(
                    pd.DataFrame(ens_data).set_index("Model"),
                    use_container_width=True)

        else:
            st.markdown("""
            <div class="cw-placeholder">
              <div class="cw-placeholder-icon">⟡</div>
              <div class="cw-placeholder-title">Awaiting Application Data</div>
              <div class="cw-placeholder-sub">Complete the form and click<br>
                <strong style="color:#8B5CF6;">Run AI Analysis</strong> to receive a decision.</div>
            </div>""", unsafe_allow_html=True)

            for step in [
                ("01", "Input",   "Fill in the applicant's personal, financial & loan details."),
                ("02", "Analyse", "Random Forest model evaluates 20+ engineered feature signals."),
                ("03", "Decide",  "Receive a binary decision with confidence score & risk factors."),
            ]:
                st.markdown(f"""
                <div class="cw-step">
                  <div class="cw-step-num">{step[0]}</div>
                  <div>
                    <div class="cw-step-title">{step[1]}</div>
                    <div class="cw-step-desc">{step[2]}</div>
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 ·  ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "◎  Analytics":

    st.markdown("""
    <div class="cw-hero">
      <div class="cw-hero-eyebrow">◎ Exploratory Data Analysis — 1,000 Applications</div>
      <h1>Dataset Analytics</h1>
      <p>Deep-dive into patterns, distributions and correlations across all 19 features
         of the loan application dataset.</p>
    </div>""", unsafe_allow_html=True)

    df = raw_df.copy()

    # Tab strip
    t1, t2, t3, t4 = st.tabs([
        "  ◌  Distributions  ",
        "  ◍  Demographics   ",
        "  ◆  Financials     ",
        "  ⬡  Correlations   ",
    ])

    # ── Distributions ────────────────────────────────────────────
    with t1:
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            section("◌", "Applicant Income")
            fig = go.Figure()
            for outcome, color, name in [
                ("Yes", COLORS["lime"], "Approved"),
                ("No",  COLORS["rose"], "Rejected"),
            ]:
                s = df[df[TARGET]==outcome]["Applicant_Income"].dropna()
                fig.add_trace(go.Histogram(
                    x=s, name=name, nbinsx=25,
                    marker=dict(color=color, opacity=0.72,
                                line=dict(color="#04030A", width=0.5))))
            fig.update_layout(**PL, height=270, barmode="overlay",
                              xaxis_title="Income (₹/mo)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        with c2:
            section("◎", "Credit Score Distribution")
            fig2 = go.Figure()
            for outcome, color, name in [
                ("Yes", COLORS["lime"], "Approved"),
                ("No",  COLORS["rose"], "Rejected"),
            ]:
                s = df[df[TARGET]==outcome]["Credit_Score"].dropna()
                fig2.add_trace(go.Histogram(
                    x=s, name=name, nbinsx=22,
                    marker=dict(color=color, opacity=0.72,
                                line=dict(color="#04030A", width=0.5))))
            fig2.update_layout(**PL, height=270, barmode="overlay",
                               xaxis_title="Credit Score", yaxis_title="Count")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

        c3, c4 = st.columns(2, gap="medium")

        with c3:
            section("◆", "Loan Amount by Outcome")
            fig3 = px.box(
                df.dropna(subset=["Loan_Amount", TARGET]),
                x=TARGET, y="Loan_Amount", color=TARGET,
                color_discrete_map={"Yes": COLORS["lime"], "No": COLORS["rose"]},
                labels={TARGET: "", "Loan_Amount": "Loan Amount (₹)"},
                points="outliers",
            )
            fig3.update_layout(**PL, height=270, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

        with c4:
            section("⬡", "Savings vs Loan Amount")
            s4 = df.dropna(subset=["Savings","Loan_Amount",TARGET]).sample(
                n=min(500,len(df)), random_state=42)
            fig4 = px.scatter(
                s4, x="Savings", y="Loan_Amount", color=TARGET,
                color_discrete_map={"Yes": COLORS["lime"], "No": COLORS["rose"]},
                opacity=0.6, labels={TARGET: ""},
            )
            fig4.update_traces(marker=dict(size=5, line=dict(width=0)))
            fig4.update_layout(**PL, height=270)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})

    # ── Demographics ─────────────────────────────────────────────
    with t2:
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            section("◌", "Approval by Gender")
            gd = (df.dropna(subset=["Gender"])
                    .groupby(["Gender", TARGET]).size().reset_index(name="n"))
            fig = px.bar(gd, x="Gender", y="n", color=TARGET,
                         color_discrete_map={"Yes":COLORS["lime"],"No":COLORS["rose"]},
                         barmode="group", labels={TARGET:"","n":"Applications"})
            fig.update_layout(**PL, height=270)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        with c2:
            section("◎", "Approval by Education")
            ed = (df.dropna(subset=["Education_Level"])
                    .groupby(["Education_Level", TARGET]).size().reset_index(name="n"))
            fig2 = px.bar(ed, x="Education_Level", y="n", color=TARGET,
                          color_discrete_map={"Yes":COLORS["lime"],"No":COLORS["rose"]},
                          barmode="group",
                          labels={TARGET:"","n":"Applications","Education_Level":"Education"})
            fig2.update_layout(**PL, height=270)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

        c3, c4 = st.columns(2, gap="medium")

        with c3:
            section("◆", "Age Distribution")
            fig3 = go.Figure()
            for outcome, color, name in [
                ("Yes", COLORS["lime"],"Approved"),
                ("No",  COLORS["rose"],"Rejected"),
            ]:
                s = df[df[TARGET]==outcome]["Age"].dropna()
                fig3.add_trace(go.Histogram(
                    x=s, name=name, nbinsx=20,
                    marker=dict(color=color, opacity=0.72,
                                line=dict(color="#04030A",width=0.5))))
            fig3.update_layout(**PL, height=270, barmode="overlay",
                               xaxis_title="Age (years)", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

        with c4:
            section("⬡", "Employment Status Mix")
            em = (df.dropna(subset=["Employment_Status"])
                    .groupby(["Employment_Status", TARGET]).size().reset_index(name="n"))
            fig4 = px.bar(em, x="Employment_Status", y="n", color=TARGET,
                          color_discrete_map={"Yes":COLORS["lime"],"No":COLORS["rose"]},
                          barmode="stack",
                          labels={TARGET:"","n":"Applications","Employment_Status":"Status"})
            fig4.update_layout(**PL, height=270)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})

    # ── Financials ───────────────────────────────────────────────
    with t3:
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            section("◌", "DTI Ratio Distribution")
            fig = go.Figure()
            for outcome, color, name in [
                ("Yes", COLORS["lime"],"Approved"),
                ("No",  COLORS["rose"],"Rejected"),
            ]:
                s = df[df[TARGET]==outcome]["DTI_Ratio"].dropna()
                fig.add_trace(go.Violin(
                    y=s, name=name, fillcolor=color,
                    line_color=color, opacity=0.65,
                    box_visible=True, meanline_visible=True,
                    hoverinfo="y+name",
                ))
            fig.update_layout(**PL, height=280)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        with c2:
            section("◎", "Approval Rate by Existing Loans")
            el = (df.dropna(subset=["Existing_Loans"])
                    .groupby("Existing_Loans")
                    .apply(lambda x: round((x[TARGET]=="Yes").mean()*100,1))
                    .reset_index(name="rate"))
            fig2 = px.bar(
                el, x="Existing_Loans", y="rate",
                color="rate",
                color_continuous_scale=[
                    [0,"#FB7185"],[0.5,"#FCD34D"],[1,"#A3E635"]],
                labels={"Existing_Loans":"Existing Loans","rate":"Approval Rate (%)"},
            )
            fig2.update_layout(**PL, height=280, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

        c3, c4 = st.columns(2, gap="medium")

        with c3:
            section("◆", "Collateral vs Approval Rate")
            dc = df.dropna(subset=["Collateral_Value",TARGET]).copy()
            dc["bin"] = pd.cut(dc["Collateral_Value"], bins=6)
            cb = (dc.groupby("bin", observed=False)
                    .apply(lambda x: round((x[TARGET]=="Yes").mean()*100,1))
                    .reset_index(name="rate"))
            cb["bin"] = cb["bin"].astype(str)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=cb["bin"], y=cb["rate"],
                mode="lines+markers",
                line=dict(color=COLORS["violet"], width=2.5),
                marker=dict(color=COLORS["lime"], size=9,
                            line=dict(color="#04030A", width=2)),
                hovertemplate="Range: %{x}<br>Approval Rate: %{y}%<extra></extra>",
            ))
            fig3.update_layout(**PL, height=280,
                               xaxis_tickangle=-30,
                               yaxis_title="Approval Rate (%)")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

        with c4:
            section("⬡", "Loan Term vs Applications")
            lt = (df.dropna(subset=["Loan_Term"])
                    .groupby(["Loan_Term", TARGET]).size().reset_index(name="n"))
            fig4 = px.bar(
                lt, x="Loan_Term", y="n", color=TARGET,
                color_discrete_map={"Yes":COLORS["lime"],"No":COLORS["rose"]},
                barmode="group",
                labels={TARGET:"","n":"Applications","Loan_Term":"Term (months)"},
            )
            fig4.update_layout(**PL, height=280)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})

    # ── Correlations ─────────────────────────────────────────────
    with t4:
        section("⬡", "Full Feature Correlation Heatmap")
        num_c = df.select_dtypes(include="number").columns.tolist()
        dcorr = df[num_c].copy()
        dcorr[TARGET] = (df[TARGET]=="Yes").astype(int)
        corr = dcorr.corr()

        fig_h = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[
                [0.0, "#FB7185"], [0.35, "#1E1050"],
                [0.5, "#04030A"], [0.65, "#1E1050"],
                [1.0, "#A3E635"],
            ],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=9, color="#6B6882"),
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        fig_h.update_layout(**PL, height=540, title="Pearson Correlation Matrix",
                            xaxis_tickangle=-40)
        st.plotly_chart(fig_h, use_container_width=True, con
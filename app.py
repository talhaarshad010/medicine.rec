"""
Medicine Recommendation System — Streamlit Web Application
===========================================================
Module 6: Monochrome (Black & White) Interactive Web UI.

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medicine Recommendation System",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Monochrome CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #000000;
    color: #ffffff;
}
.main .block-container {
    padding: 2rem 1.5rem 4rem;
    max-width: 720px;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}

/* ── Logo mark ── */
.logo-mark {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.2rem;
}
.logo-icon {
    width: 38px; height: 38px;
    background: #ffffff;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; color: #000000;
    flex-shrink: 0;
}
.logo-text {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
}

/* ── Hero ── */
.hero-sub {
    font-size: 0.88rem;
    color: #888888;
    margin: 0.1rem 0 1.8rem 0;
    line-height: 1.5;
}

/* ── Pill tags ── */
.symptom-pills {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-bottom: 1.2rem;
}
.pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    background: #111111;
    color: #ffffff;
    border: 1px solid #333333;
}

/* ── Result section ── */
.result-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #666666;
    margin-bottom: 0.35rem;
}
.disease-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}

/* ── Severity meter ── */
.sev-bar-track {
    width: 100%;
    height: 4px;
    background: #222222;
    border-radius: 0;
    overflow: hidden;
    margin-top: 0.4rem;
}
.sev-bar-fill {
    height: 100%;
    background: #ffffff;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.sev-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.35rem;
}
.sev-score {
    font-size: 0.78rem;
    font-weight: 600;
    color: #888888;
}
.sev-level {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 2px 8px;
    border: 1px solid #ffffff;
    color: #ffffff;
}

/* ── Divider ── */
.subtle-divider {
    border: none;
    border-top: 1px solid #222222;
    margin: 1.4rem 0;
}

/* ── Description ── */
.desc-text {
    font-size: 0.9rem;
    line-height: 1.75;
    color: #cccccc;
}

/* ── Precaution cards ── */
.prec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.prec-card {
    background: #000000;
    border: 1px solid #222222;
    border-radius: 0;
    padding: 1.2rem;
    transition: border-color 0.2s ease;
}
.prec-card:hover {
    border-color: #ffffff;
}
.prec-num {
    font-size: 0.65rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.3rem;
    opacity: 0.5;
}
.prec-text {
    font-size: 0.82rem;
    color: #ffffff;
    line-height: 1.45;
    text-transform: capitalize;
}

/* ── Chart section ── */
.chart-section {
    margin-top: 0.5rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    border: 1px dashed #222222;
    margin-bottom: 2rem;
}
.empty-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    filter: grayscale(100%);
}
.empty-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.4rem;
}
.empty-desc {
    font-size: 0.82rem;
    color: #666666;
    line-height: 1.6;
}

/* ── Stat boxes on landing ── */
.stat-row {
    display: flex; gap: 12px;
}
.stat-box {
    flex: 1;
    background: #000000;
    border: 1px solid #222222;
    padding: 1.2rem 0.8rem;
    text-align: center;
}
.stat-num {
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
}
.stat-label {
    font-size: 0.68rem;
    font-weight: 500;
    color: #666666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.15rem;
}

/* ── Streamlit overrides ── */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 0 !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #000000 !important;
}
.stMultiSelect [data-baseweb="select"] {
    border-radius: 0 !important;
    border-color: #222222 !important;
    background-color: #000000 !important;
}
.stButton > button {
    width: 100%;
    background: #ffffff;
    color: #000000;
    border: none;
    border-radius: 0;
    padding: 0.75rem 0;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.5px;
    transition: opacity 0.2s ease;
}
.stButton > button:hover {
    background: #ffffff;
    opacity: 0.8;
    color: #000000;
}
.stButton > button:active {
    background: #ffffff;
    transform: translateY(1px);
}
</style>
""", unsafe_allow_html=True)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf           = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    all_symptoms = joblib.load(os.path.join(MODEL_DIR, "symptoms_list.pkl"))
    sev_dict     = joblib.load(os.path.join(MODEL_DIR, "severity_dict.pkl"))
    df_desc = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
    df_prec = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
    for df in [df_desc, df_prec]:
        df.columns = df.columns.str.strip()
        df["Disease"] = df["Disease"].str.strip()
    df_desc["Description"] = df_desc["Description"].str.strip()
    return rf, le, all_symptoms, sev_dict, df_desc, df_prec


def predict(symptoms_input, rf, le, all_symptoms, sev_dict, df_desc, df_prec):
    symptom_to_idx = {s: i for i, s in enumerate(all_symptoms)}
    x = np.zeros(len(all_symptoms), dtype=np.float32)
    severity_score = 0
    for sym in symptoms_input:
        sym = sym.strip()
        if sym in symptom_to_idx:
            w = sev_dict.get(sym, 1)
            x[symptom_to_idx[sym]] = w
            severity_score += w

    pred_idx = rf.predict(x.reshape(1, -1))[0]
    disease = le.inverse_transform([pred_idx])[0]

    row_d = df_desc[df_desc["Disease"] == disease]
    description = row_d["Description"].values[0] if not row_d.empty else "Description not available."

    row_p = df_prec[df_prec["Disease"] == disease]
    precautions = []
    if not row_p.empty:
        for i in range(1, 5):
            v = row_p[f"Precaution_{i}"].values[0]
            if pd.notna(v) and str(v).strip():
                precautions.append(str(v).strip())
    if not precautions:
        precautions = ["Precautions not available."]

    imps = rf.feature_importances_
    top10 = np.argsort(imps)[::-1][:10]
    imp_df = pd.DataFrame({
        "Importance": [imps[i] for i in top10],
    }, index=[all_symptoms[i] for i in top10])

    return {
        "disease": disease,
        "description": description,
        "precautions": precautions,
        "severity_score": int(severity_score),
        "importances_df": imp_df,
    }


# ─── UI ───────────────────────────────────────────────────────────────────────

try:
    rf, le, all_symptoms, sev_dict, df_desc, df_prec = load_model()
except FileNotFoundError:
    st.error("Model files not found. Run `python medicine_recommendation.py` first.")
    st.stop()

# Header
st.markdown("""
<div class="logo-mark">
    <div class="logo-icon">M</div>
    <div class="logo-text">MEDICINE.REC</div>
</div>
<p class="hero-sub">
    MONOCHROME DIAGNOSTIC INTERFACE<br>
    SYMPTOM ANALYSIS & DISEASE PREDICTION ENGINE
</p>
""", unsafe_allow_html=True)

# Symptom selector
selected = st.multiselect(
    "SELECT SYMPTOMS",
    options=all_symptoms,
    default=[],
    placeholder="SEARCH...",
)

# Action button
clicked = st.button("EXECUTE ANALYSIS", use_container_width=True)

# ─── Results ──────────────────────────────────────────────────────────────────
if clicked:
    if not selected:
        st.warning("PLEASE SELECT AT LEAST ONE SYMPTOM.")
    else:
        result = predict(selected, rf, le, all_symptoms, sev_dict, df_desc, df_prec)
        sev = result["severity_score"]
        pct = min(sev / 70 * 100, 100)

        # Selected symptoms as pills
        pills = "".join(f'<span class="pill">{s.upper()}</span>' for s in selected)
        st.markdown(f'<div class="symptom-pills">{pills}</div>', unsafe_allow_html=True)

        # Disease name
        st.markdown(f"""
        <div class="result-label">DIAGNOSIS</div>
        <div class="disease-title">{result['disease'].upper()}</div>
        """, unsafe_allow_html=True)

        # Severity bar
        st.markdown(f"""
        <div class="sev-bar-track">
            <div class="sev-bar-fill" style="width:{pct:.1f}%;"></div>
        </div>
        <div class="sev-meta">
            <span class="sev-score">{sev} / 70</span>
            <span class="sev-level">SEVERITY LEVEL</span>
        </div>
        """, unsafe_allow_html=True)

        # Divider
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Description
        st.markdown(f"""
        <div class="result-label">CLINICAL DESCRIPTION</div>
        <p class="desc-text">{result['description']}</p>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Precautions
        st.markdown('<div class="result-label">ACTIONABLE PRECAUTIONS</div>', unsafe_allow_html=True)
        prec_cards = ""
        for i, p in enumerate(result["precautions"], 1):
            prec_cards += f"""
            <div class="prec-card">
                <div class="prec-num">PROTOCOL {i:02d}</div>
                <div class="prec-text">{p}</div>
            </div>"""
        st.markdown(f'<div class="prec-grid">{prec_cards}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Feature importance chart
        st.markdown('<div class="result-label">MODEL WEIGHTS (TOP 10)</div>', unsafe_allow_html=True)
        st.bar_chart(result["importances_df"], horizontal=True, color="#ffffff")

elif not selected:
    # Empty / landing state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⌘</div>
        <div class="empty-title">SYSTEM IDLE</div>
        <div class="empty-desc">
            INITIALIZE BY SELECTING SYMPTOMS FROM THE DROPDOWN ABOVE.<br>
            THEN EXECUTE ANALYSIS TO RETRIEVE DIAGNOSTIC DATA.
        </div>
    </div>
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-num">41</div>
            <div class="stat-label">DISEASES</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">131</div>
            <div class="stat-label">SYMPTOMS</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">4,920</div>
            <div class="stat-label">CASES</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">100%</div>
            <div class="stat-label">ACCURACY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

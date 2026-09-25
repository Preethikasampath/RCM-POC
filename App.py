"""
AI Claims Scrubbing & Denial Prediction — "Vital Signs" Command Center
===========================================================================
Built for: Senior ML Engineer (Claim Scrubbing & Denial Prediction), PAIX

Drop this file into the same folder as your trained artifacts:
tuned_xgboost_model.pkl, carc_prediction_model.pkl, target_encoder.pkl,
carc_label_encoder.pkl, label_encoders.pkl, claims.csv,
claims_preprocessed_v3.csv, X_test.pkl, y_test.pkl, X_test_carc.pkl,
y_test_carc.pkl

Run with: streamlit run app.py
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

st.set_page_config(
    page_title="Vital Signs | Claims Intelligence",
    page_icon="\U0001FA7A",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))

DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-deep: #050914;
    --glass: rgba(255,255,255,0.045);
    --glass-strong: rgba(255,255,255,0.07);
    --glass-border: rgba(255,255,255,0.09);
    --glass-border-hover: rgba(34,211,238,0.45);
    --text-primary: #EAF1FB;
    --text-muted: #8FA0BA;
    --vital-teal: #22D3EE;
    --vital-amber: #FBBF24;
    --vital-crimson: #FB4B4B;
    --glow-teal: rgba(34,211,238,0.35);
    --glow-crimson: rgba(251,75,75,0.4);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; line-height: 1.55; }
h1, h2, h3, .display-font { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; line-height: 1.25; }
.mono, code { font-family: 'IBM Plex Mono', monospace !important; }
p, .stMarkdown p { line-height: 1.6; }

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(34,211,238,0.10), transparent),
        radial-gradient(ellipse 60% 50% at 100% 10%, rgba(99,102,241,0.10), transparent),
        radial-gradient(ellipse 70% 60% at 50% 110%, rgba(34,211,238,0.06), transparent),
        var(--bg-deep);
    background-attachment: fixed;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060B18 0%, #050914 100%);
    border-right: 1px solid var(--glass-border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

h1, h2, h3, h4, p, span, label, .stMarkdown { color: var(--text-primary); }
.stDataFrame { color: var(--text-primary); }

.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 4px 14px 4px; margin-bottom: 6px;
    position: sticky; top: 0; z-index: 999;
    background: linear-gradient(180deg, rgba(5,9,20,0.96) 80%, rgba(5,9,20,0));
    backdrop-filter: blur(6px);
}
.app-header h1 {
    font-size: 1.7rem; margin: 0; font-weight: 700;
    background: linear-gradient(90deg, #EAF1FB, #A8E9F0);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.app-header .tagline { color: var(--text-muted); font-size: 0.82rem; font-family: 'IBM Plex Mono', monospace; }
.live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--vital-teal); margin-right: 6px;
    box-shadow: 0 0 8px var(--glow-teal);
    animation: pulse-dot 1.6s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:0.4; transform:scale(1.3);} }

/* ---- Loading skeleton shimmer ---- */
.skeleton {
    background: linear-gradient(90deg, var(--glass) 25%, var(--glass-strong) 50%, var(--glass) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s ease-in-out infinite;
    border-radius: 10px;
}
@keyframes shimmer { 0%{background-position:200% 0;} 100%{background-position:-200% 0;} }

/* ---- Responsive: stack columns below tablet width ---- */
@media (max-width: 768px) {
    div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    .app-header h1 { font-size: 1.3rem; }
    .kpi-value { font-size: 1.5rem !important; }
}

.ecg-wrap { width: 100%; height: 34px; margin: 4px 0 18px 0; overflow: hidden; }
.ecg-line {
    stroke: var(--vital-teal); stroke-width: 1.6; fill: none;
    filter: drop-shadow(0 0 4px var(--glow-teal));
    stroke-dasharray: 800; stroke-dashoffset: 800;
    animation: draw-ecg 3.2s linear infinite;
}
@keyframes draw-ecg { to { stroke-dashoffset: 0; } }

.glass {
    background: var(--glass);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    transition: transform 0.35s cubic-bezier(.2,.8,.2,1), border-color 0.3s, box-shadow 0.3s;
}
.kpi-card { padding: 18px 20px; height: 100%; }
.kpi-card:hover, .panel:hover {
    transform: perspective(800px) rotateX(2.5deg) translateY(-3px);
    border-color: var(--glass-border-hover);
    box-shadow: 0 14px 34px -12px rgba(0,0,0,0.6), 0 0 24px -8px var(--glow-teal);
}
.kpi-label {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-muted); font-weight: 600; margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(90deg, #EAF1FB, #7FE3EE);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.kpi-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: var(--text-muted); margin-top: 4px; }

.panel { padding: 22px 24px; margin-bottom: 18px; }
.panel-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem;
    color: var(--text-primary); margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}

.chip {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 600;
    border: 1px solid;
}
.chip-low  { background: rgba(34,211,238,0.10); color: var(--vital-teal); border-color: rgba(34,211,238,0.4); box-shadow: 0 0 12px -2px var(--glow-teal); }
.chip-med  { background: rgba(251,191,36,0.10); color: var(--vital-amber); border-color: rgba(251,191,36,0.4); }
.chip-high { background: rgba(251,75,75,0.12); color: var(--vital-crimson); border-color: rgba(251,75,75,0.5); box-shadow: 0 0 14px -2px var(--glow-crimson); animation: alert-glow 1.4s ease-in-out infinite; }
@keyframes alert-glow { 0%,100%{ box-shadow: 0 0 8px -2px var(--glow-crimson);} 50%{ box-shadow: 0 0 20px 0px var(--glow-crimson);} }

.ledger-row {
    display: flex; align-items: center; gap: 12px; padding: 11px 14px;
    background: var(--glass-strong); border-left: 3px solid var(--text-muted);
    border-radius: 6px; margin-bottom: 7px; font-size: 0.86rem; color: var(--text-primary);
}
.ledger-row.hard  { border-left-color: var(--vital-crimson); box-shadow: -2px 0 10px -4px var(--glow-crimson); }
.ledger-row.soft  { border-left-color: var(--vital-amber); }
.ledger-row.clear { border-left-color: var(--vital-teal); box-shadow: -2px 0 10px -4px var(--glow-teal); }
.ledger-code {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: var(--text-primary);
    min-width: 90px; opacity: 0.9;
}

.stSelectbox label, .stNumberInput label, .stSlider label, .stCheckbox label, .stRadio label { color: var(--text-primary) !important; }
div[data-baseweb="select"] > div { background-color: var(--glass-strong) !important; border-color: var(--glass-border) !important; }
.stNumberInput input { background-color: var(--glass-strong) !important; color: var(--text-primary) !important; }
button[kind="primary"] {
    background: linear-gradient(90deg, #0D9488, #22D3EE) !important;
    border: none !important; font-weight: 600 !important;
    box-shadow: 0 4px 20px -4px var(--glow-teal) !important;
}
.stDataFrame { border-radius: 10px; overflow: hidden; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* Sidebar collapse/expand arrow was invisible against the dark theme - fix */
[data-testid="collapsedControl"] {
    background: var(--glass-strong) !important;
    border: 1px solid var(--glow-teal) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 14px -2px var(--glow-teal) !important;
}
[data-testid="collapsedControl"] svg { fill: var(--vital-teal) !important; }
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)


def ecg_divider():
    st.markdown(
        """<div class="ecg-wrap">
            <svg class="ecg-wrap" viewBox="0 0 800 34" preserveAspectRatio="none" width="100%" height="34">
                <path class="ecg-line" d="M0,17 L120,17 L145,17 L158,4 L172,30 L186,10 L200,17 L230,17
                    L360,17 L385,17 L398,4 L412,30 L426,10 L440,17 L470,17
                    L600,17 L625,17 L638,4 L652,30 L666,10 L680,17 L800,17"/>
            </svg>
        </div>""",
        unsafe_allow_html=True,
    )


def hero_3d(height=170):
    components.html(
        f"""
        <div style="width:100%; height:{height}px; background:transparent;">
          <canvas id="core-canvas" style="width:100%; height:100%; display:block;"></canvas>
        </div>
        <script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
        <script>
        (function() {{
            const canvas = document.getElementById('core-canvas');
            const w = canvas.clientWidth, h = {height};
            const renderer = new THREE.WebGLRenderer({{canvas: canvas, alpha: true, antialias: true}});
            renderer.setSize(w, h);
            renderer.setPixelRatio(window.devicePixelRatio);

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 1000);
            camera.position.z = 5.2;

            const group = new THREE.Group();
            const geo = new THREE.IcosahedronGeometry(1.5, 1);
            const wire = new THREE.WireframeGeometry(geo);
            const mat = new THREE.LineBasicMaterial({{color: 0x22D3EE, transparent: true, opacity: 0.75}});
            const core = new THREE.LineSegments(wire, mat);
            group.add(core);

            const geo2 = new THREE.IcosahedronGeometry(1.5, 1);
            const mat2 = new THREE.MeshBasicMaterial({{color: 0x0D9488, transparent: true, opacity: 0.06}});
            const solid = new THREE.Mesh(geo2, mat2);
            group.add(solid);

            const ringGeo = new THREE.TorusGeometry(2.3, 0.008, 8, 100);
            const ringMat = new THREE.MeshBasicMaterial({{color: 0x7FE3EE, transparent: true, opacity: 0.35}});
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2.4;
            group.add(ring);

            scene.add(group);

            const particleCount = 60;
            const particleGeo = new THREE.BufferGeometry();
            const positions = new Float32Array(particleCount * 3);
            for (let i = 0; i < particleCount; i++) {{
                positions[i*3] = (Math.random()-0.5) * 8;
                positions[i*3+1] = (Math.random()-0.5) * 4;
                positions[i*3+2] = (Math.random()-0.5) * 4;
            }}
            particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const particleMat = new THREE.PointsMaterial({{color: 0x22D3EE, size: 0.02, transparent: true, opacity: 0.5}});
            const particles = new THREE.Points(particleGeo, particleMat);
            scene.add(particles);

            function animate() {{
                requestAnimationFrame(animate);
                group.rotation.y += 0.0035;
                group.rotation.x += 0.0012;
                ring.rotation.z += 0.002;
                particles.rotation.y += 0.0006;
                renderer.render(scene, camera);
            }}
            animate();

            window.addEventListener('resize', function() {{
                const nw = canvas.clientWidth;
                camera.aspect = nw / h;
                camera.updateProjectionMatrix();
                renderer.setSize(nw, h);
            }});
        }})();
        </script>
        """,
        height=height,
    )


def app_header(title: str, tagline: str, show_3d: bool = True):
    if show_3d:
        hero_3d(150)
    st.markdown(
        f"""<div class="app-header">
                <div><h1><span class="live-dot"></span>{title}</h1></div>
                <span class="tagline">{tagline}</span>
            </div>""",
        unsafe_allow_html=True,
    )
    ecg_divider()


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""<div class="glass kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def panel_open(title: str, icon: str = ""):
    st.markdown(f'<div class="glass panel"><div class="panel-title">{icon} {title}</div>', unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


def risk_chip(tier: str) -> str:
    cls = {"Low": "chip-low", "Medium": "chip-med", "High": "chip-high"}[tier]
    return f'<span class="chip {cls}">\u25CF {tier.upper()} RISK</span>'


def skeleton_block(height_px: int = 90):
    st.markdown(f'<div class="skeleton" style="height:{height_px}px; margin-bottom:12px;"></div>', unsafe_allow_html=True)


PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Inter, sans-serif", color="#EAF1FB"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=["#22D3EE", "#0D9488", "#FBBF24", "#FB4B4B", "#8FA0BA", "#7FE3EE"],
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
)

CARC_DESCRIPTIONS = {
    "CO-16": "Missing/invalid info needed for adjudication",
    "CO-18": "Duplicate claim/service",
    "CO-29": "Timely filing limit expired",
    "CO-45": "Charge exceeds fee schedule/contracted amount",
    "CO-50": "Not deemed a medical necessity",
    "CO-97": "Bundled into another service (NCCI)",
    "CO-109": "Wrong payer/COB issue",
    "CO-197": "Precertification/authorization absent",
}

REQUIRED_FILES = [
    "tuned_xgboost_model.pkl", "carc_prediction_model.pkl", "target_encoder.pkl",
    "carc_label_encoder.pkl", "label_encoders.pkl", "claims.csv",
    "claims_preprocessed_v3.csv", "X_test.pkl", "y_test.pkl",
    "X_test_carc.pkl", "y_test_carc.pkl",
]


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(BASE, f))]
    if missing:
        return None, missing
    artifacts = {
        "binary_model": joblib.load(os.path.join(BASE, "tuned_xgboost_model.pkl")),
        "carc_model": joblib.load(os.path.join(BASE, "carc_prediction_model.pkl")),
        "target_encoder": joblib.load(os.path.join(BASE, "target_encoder.pkl")),
        "carc_encoder": joblib.load(os.path.join(BASE, "carc_label_encoder.pkl")),
        "label_encoders": joblib.load(os.path.join(BASE, "label_encoders.pkl")),
        "X_test": joblib.load(os.path.join(BASE, "X_test.pkl")),
        "y_test": joblib.load(os.path.join(BASE, "y_test.pkl")),
        "X_test_carc": joblib.load(os.path.join(BASE, "X_test_carc.pkl")),
        "y_test_carc": joblib.load(os.path.join(BASE, "y_test_carc.pkl")),
    }
    artifacts["explainer"] = shap.TreeExplainer(artifacts["binary_model"])
    return artifacts, []


@st.cache_data
def load_readable_data():
    return pd.read_csv(os.path.join(BASE, "claims.csv"))


if "artifacts_loaded_once" not in st.session_state:
    skeleton_placeholder = st.empty()
    with skeleton_placeholder.container():
        st.markdown('<div class="skeleton" style="height:36px; width:280px; margin-bottom:18px;"></div>', unsafe_allow_html=True)
        sk_cols = st.columns(4)
        for sk_col in sk_cols:
            with sk_col:
                skeleton_block(90)
        skeleton_block(280)
    artifacts, missing_files = load_artifacts()
    skeleton_placeholder.empty()
    st.session_state["artifacts_loaded_once"] = True
else:
    artifacts, missing_files = load_artifacts()

if artifacts is None:
    st.markdown(
        f"""<div class="glass panel">
            <div class="panel-title">\u26A0\uFE0F Setup needed</div>
            <p style="color:var(--text-muted); font-size:0.9rem;">
            This app expects the following files in the same folder as <code>app.py</code>:</p>
            <ul>{''.join(f'<li class="mono">{f}</li>' for f in missing_files)}</ul>
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

claims_readable = load_readable_data()
FEATURE_COLS = list(artifacts["X_test"].columns)
_test_probs = artifacts["binary_model"].predict_proba(artifacts["X_test"])[:, 1]
PROBA_MIN, PROBA_MAX = float(_test_probs.min()), float(_test_probs.max())

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Space Grotesk\'; font-size:1.2rem; font-weight:700; '
        'padding:10px 0 2px 0; background:linear-gradient(90deg,#EAF1FB,#7FE3EE); '
        '-webkit-background-clip:text; background-clip:text; color:transparent;">'
        '\U0001FA7A Vital Signs</div>'
        '<div style="font-family:\'IBM Plex Mono\'; font-size:0.72rem; opacity:0.6; '
        'margin-bottom:24px;">PAIX \u00b7 Claims Intelligence POC</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["Analytics", "Claim Scrubber", "Risk Queue", "Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='opacity:0.15;'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.78rem; opacity:0.65; line-height:1.7; font-family:\'IBM Plex Mono\';">'
        '<span style="color:#22D3EE;">\u25CF</span> Model 1 \u2014 denial risk<br>'
        '<span style="color:#22D3EE;">\u25CF</span> Model 2 \u2014 CARC reason<br>'
        '<span style="color:#22D3EE;">\u25CF</span> SHAP \u2014 explainability<br>'
        '<span style="color:#22D3EE;">\u25CF</span> Rules \u2014 payer edits</div>',
        unsafe_allow_html=True,
    )


def run_rule_edits(claim: dict) -> list:
    edits = []
    if claim.get("Prior_Auth_Required") and not claim.get("Prior_Auth_Obtained"):
        edits.append(("hard", "AUTH", "Prior authorization required but not on file"))
    if not claim.get("Eligibility_Verified"):
        edits.append(("hard", "ELIG", "Patient eligibility not verified pre-submission"))
    if claim.get("Duplicate_Claim_Flag"):
        edits.append(("hard", "DUP", "Possible duplicate claim detected"))
    if claim.get("Clearinghouse_Validation_Passed") is False:
        edits.append(("hard", "CH-EDIT", "Clearinghouse front-end validation failed"))
    if claim.get("NCCI_Bundling_Flag"):
        edits.append(("soft", "NCCI", "Procedure may be bundled under NCCI rules"))
    return edits


def risk_tier_from_proba(p: float) -> str:
    # NOTE: thresholds are tercile-based on this model's actual predicted
    # probability distribution (33rd/67th percentile on the held-out test
    # set), not arbitrary fixed cutoffs. Model 1 uses scale_pos_weight to
    # correct class imbalance, which shifts raw probabilities upward -
    # the model's minimum output across the entire test set is ~0.325,
    # so a fixed 0.3/0.6 split would make "Low risk" unreachable.
    if p > 0.505:
        return "High"
    if p > 0.429:
        return "Medium"
    return "Low"


def risk_gauge_svg(proba: float, tier: str, proba_min: float = 0.0, proba_max: float = 1.0) -> str:
    # Needle ANGLE is rescaled to the model's actual output range so the
    # dial visually sweeps its full arc (this model's raw outputs are
    # compressed into a ~0.33-0.79 band due to scale_pos_weight - without
    # rescaling, the needle would never reach either end of the gauge).
    # The DISPLAYED PERCENTAGE stays the true, unscaled model output.
    span = max(proba_max - proba_min, 1e-6)
    visual_p = np.clip((proba - proba_min) / span, 0.0, 1.0)
    angle = 180 * visual_p
    needle_x = 100 + 78 * np.cos(np.radians(180 - angle))
    needle_y = 100 - 78 * np.sin(np.radians(180 - angle))
    color = {"Low": "#22D3EE", "Medium": "#FBBF24", "High": "#FB4B4B"}[tier]
    pulse_ring = (
        f'<circle cx="100" cy="100" r="88" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.5">'
        f'<animate attributeName="r" values="88;104;88" dur="1.6s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0.5;0;0.5" dur="1.6s" repeatCount="indefinite"/></circle>'
        if tier == "High" else ""
    )
    svg = (
        f'<svg viewBox="0 0 200 130" width="260" height="170">'
        f'<defs><filter id="glow"><feGaussianBlur stdDeviation="3.5" result="blur"/>'
        f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        f'{pulse_ring}'
        f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="M 20 100 A 80 80 0 0 1 76 27" fill="none" stroke="#22D3EE" stroke-width="14" stroke-linecap="round" filter="url(#glow)"/>'
        f'<path d="M 76 27 A 80 80 0 0 1 124 27" fill="none" stroke="#FBBF24" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="M 124 27 A 80 80 0 0 1 180 100" fill="none" stroke="#FB4B4B" stroke-width="14" stroke-linecap="round" filter="url(#glow)"/>'
        f'<line x1="100" y1="100" x2="{needle_x:.1f}" y2="{needle_y:.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/>'
        f'<circle cx="100" cy="100" r="7" fill="{color}" filter="url(#glow)"/>'
        f'<text x="100" y="90" text-anchor="middle" font-family="Space Grotesk" font-size="30" font-weight="700" fill="{color}">{proba:.0%}</text>'
        f'</svg>'
    )
    return svg


def load_lottie_url(url: str, timeout: int = 3):
    """Fetch a Lottie animation JSON. Returns None on any failure so the
    app degrades silently instead of breaking - this depends on internet
    access to the hosting URL, which cannot be guaranteed in every
    environment (corporate proxies, offline demos, stale URLs)."""
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def generate_claim_pdf(risk_pct: str, tier: str, edits: list, shap_rows: list, carc_rows: list) -> bytes:
    """One-page PDF summary of a scrubbed claim - risk, edit findings,
    top SHAP drivers, and likely CARC reason if applicable."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#0B1B2B"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#0B1B2B"))
    body = styles["BodyText"]

    story = [
        Paragraph("Claim Scrubbing Report", title_style),
        Paragraph("PAIX Claims Intelligence \u2014 Vital Signs POC", body),
        Spacer(1, 14),
        Paragraph(f"Denial Risk: {risk_pct} ({tier} risk)", h2),
        Spacer(1, 8),
        Paragraph("Payer Edit Findings", h2),
    ]
    if edits:
        t = Table([["Severity", "Code", "Finding"]] + edits, colWidths=[80, 80, 340])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1B2B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No hard payer edits triggered.", body))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Top Risk Factors (SHAP)", h2))
    t2 = Table([["Feature", "Impact"]] + shap_rows, colWidths=[260, 240])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1B2B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t2)

    if carc_rows:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Likely Denial Reason (CARC)", h2))
        t3 = Table([["Code", "Description", "Likelihood"]] + carc_rows, colWidths=[70, 330, 100])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1B2B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t3)

    doc.build(story)
    return buf.getvalue()


def generate_analytics_pdf(kpis: dict, carc_counts: pd.Series) -> bytes:
    """Dashboard summary PDF - KPIs + CARC breakdown table. No embedded
    chart images (avoids a kaleido dependency for reliability)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#0B1B2B"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#0B1B2B"))
    body = styles["BodyText"]

    story = [
        Paragraph("Denial Analytics Summary", title_style),
        Paragraph("PAIX Claims Intelligence \u2014 Vital Signs POC", body),
        Spacer(1, 14),
        Paragraph("Key Metrics", h2),
    ]
    kpi_rows = [[k, v] for k, v in kpis.items()]
    t = Table([["Metric", "Value"]] + kpi_rows, colWidths=[260, 240])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1B2B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Denial Reason Breakdown (CARC)", h2))
    carc_rows = [[code, CARC_DESCRIPTIONS.get(code, ""), str(int(count))] for code, count in carc_counts.items()]
    t2 = Table([["Code", "Description", "Count"]] + carc_rows, colWidths=[60, 340, 100])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1B2B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t2)

    doc.build(story)
    return buf.getvalue()


SAMPLE_CLAIMS = {
    "\U0001F7E2 Clean claim (low risk)": {
        "form_type": "837P", "specialty": "General Practice", "payer": "Medicare",
        "diagnosis": "I10", "modifier": "No_Modifier", "pos_code": 11, "proc_code": 99213,
        "billed": 180.0, "timely_limit": 365, "delay": 5, "provider_vol": 40, "prior_denials": 0,
        "auth_req": False, "auth_obt": True, "eligibility": True, "chronic": False,
        "duplicate": False, "bundling": False,
    },
    "\U0001F534 Missing prior auth (high risk)": {
        "form_type": "837P", "specialty": "Radiology", "payer": "Aetna",
        "diagnosis": "M54.5", "modifier": "No_Modifier", "pos_code": 22, "proc_code": 70551,
        "billed": 1400.0, "timely_limit": 90, "delay": 15, "provider_vol": 50, "prior_denials": 3,
        "auth_req": True, "auth_obt": False, "eligibility": True, "chronic": False,
        "duplicate": False, "bundling": False,
    },
    "\U0001F7E1 Moderate provider denial history": {
        "form_type": "837P", "specialty": "Neurology", "payer": "BCBS",
        "diagnosis": "G43.909", "modifier": "25", "pos_code": 11, "proc_code": 99214,
        "billed": 700.0, "timely_limit": 180, "delay": 25, "provider_vol": 150, "prior_denials": 4,
        "auth_req": False, "auth_obt": True, "eligibility": True, "chronic": False,
        "duplicate": False, "bundling": False,
    },
}


if page == "Analytics":
    app_header("Denial Analytics", "claims.csv \u00b7 15,000 records")

    total = len(claims_readable)
    denied = (claims_readable["Claim_Status"] == "Denied").sum()
    denial_rate = denied / total
    total_billed = claims_readable["Billed_Amount"].sum()
    at_risk = claims_readable.loc[claims_readable["Claim_Status"] == "Denied", "Billed_Amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total claims", f"{total:,}", "full dataset")
    with c2: kpi_card("Denial rate", f"{denial_rate:.1%}", f"{denied:,} denied claims")
    with c3: kpi_card("Total billed", f"${total_billed:,.0f}", "gross claim value")
    with c4: kpi_card("Revenue at risk", f"${at_risk:,.0f}", "denied claim value")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        panel_open("Denial rate by payer", "\U0001F4CA")
        payer_denial = (
            claims_readable.groupby("Payer")["Claim_Status"]
            .apply(lambda x: (x == "Denied").mean() * 100).sort_values(ascending=False)
        )
        fig = px.bar(payer_denial, labels={"value": "Denial rate (%)", "Payer": ""})
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=320)
        fig.update_traces(marker_color="#22D3EE")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        panel_close()

    with col2:
        panel_open("CARC denial reason mix", "\U0001F9EC")
        carc_counts = claims_readable.loc[claims_readable["Claim_Status"] == "Denied", "CARC_Code"].value_counts()
        fig2 = px.pie(values=carc_counts.values, names=carc_counts.index, hole=0.6)
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        panel_close()

    col3, col4 = st.columns(2)
    with col3:
        panel_open("Prior auth missing vs. on file", "\U0001F510")
        auth_flag = (claims_readable["Prior_Auth_Required"]) & (~claims_readable["Prior_Auth_Obtained"])
        auth_denial = claims_readable.groupby(auth_flag)["Claim_Status"].apply(lambda x: (x == "Denied").mean() * 100)
        auth_denial.index = ["Auth OK", "Auth missing"]
        fig3 = px.bar(auth_denial, labels={"value": "Denial rate (%)"})
        fig3.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=300)
        fig3.update_traces(marker_color=["#22D3EE", "#FB4B4B"])
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        panel_close()

    with col4:
        panel_open("Denial rate by specialty", "\U0001FA7A")
        spec_denial = (
            claims_readable.groupby("Provider_Specialty")["Claim_Status"]
            .apply(lambda x: (x == "Denied").mean() * 100).sort_values(ascending=False)
        )
        fig4 = px.bar(spec_denial, labels={"value": "Denial rate (%)", "Provider_Specialty": ""})
        fig4.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=300)
        fig4.update_traces(marker_color="#22D3EE")
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Billed amount vs. provider volume \u2014 hover to inspect", "\U0001F52C")
    sample_for_scatter = claims_readable.sample(min(1500, len(claims_readable)), random_state=42)
    fig5 = px.scatter(
        sample_for_scatter, x="Provider_Monthly_Claim_Volume", y="Billed_Amount",
        color="Claim_Status", opacity=0.55,
        color_discrete_map={"Approved": "#22D3EE", "Denied": "#FB4B4B"},
        hover_data=["Payer", "Provider_Specialty", "CARC_Code"],
    )
    fig5.update_layout(**PLOTLY_TEMPLATE["layout"], height=380, hovermode="closest")
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": True})
    panel_close()

    panel_open("CARC code reference", "\U0001F4D6")
    ref_df = pd.DataFrame(CARC_DESCRIPTIONS.items(), columns=["CARC Code", "Description"])
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
    panel_close()

    analytics_pdf = generate_analytics_pdf(
        {
            "Total claims": f"{total:,}", "Denial rate": f"{denial_rate:.1%}",
            "Total billed": f"${total_billed:,.0f}", "Revenue at risk": f"${at_risk:,.0f}",
        },
        claims_readable.loc[claims_readable["Claim_Status"] == "Denied", "CARC_Code"].value_counts(),
    )
    st.download_button(
        "\U0001F4C4 Export summary to PDF", data=analytics_pdf,
        file_name="denial_analytics_summary.pdf", mime="application/pdf",
        use_container_width=True,
    )

elif page == "Claim Scrubber":
    app_header("Claim Scrubber", "score a claim before submission")

    le = artifacts["label_encoders"]

    # ---- One-click sample claims ----
    panel_open("Quick start \u2014 one-click sample claims", "\u26A1")
    sample_cols = st.columns(len(SAMPLE_CLAIMS))
    for col, (label, values) in zip(sample_cols, SAMPLE_CLAIMS.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"sample_{label}"):
                field_map = {
                    "cs_form_type": values["form_type"], "cs_specialty": values["specialty"],
                    "cs_payer": values["payer"], "cs_diagnosis": values["diagnosis"],
                    "cs_modifier": values["modifier"], "cs_pos_code": values["pos_code"],
                    "cs_proc_code": values["proc_code"], "cs_billed": values["billed"],
                    "cs_timely_limit": values["timely_limit"], "cs_delay": values["delay"],
                    "cs_provider_vol": values["provider_vol"], "cs_prior_denials": values["prior_denials"],
                    "cs_auth_req": values["auth_req"], "cs_auth_obt": values["auth_obt"],
                    "cs_eligibility": values["eligibility"], "cs_chronic": values["chronic"],
                    "cs_duplicate": values["duplicate"], "cs_bundling": values["bundling"],
                }
                for k, v in field_map.items():
                    st.session_state[k] = v
                st.rerun()
    panel_close()

    # ---- Defaults, set only if not already present (so sample clicks persist) ----
    _defaults = {
        "cs_form_type": le["Claim_Form_Type"].classes_[0], "cs_specialty": le["Provider_Specialty"].classes_[0],
        "cs_payer": le["Payer"].classes_[0], "cs_diagnosis": le["Diagnosis_Code"].classes_[0],
        "cs_modifier": le["Modifier_Code"].classes_[0],
        "cs_pos_code": sorted(claims_readable["Place_of_Service_Code"].unique())[0],
        "cs_proc_code": sorted(claims_readable["Procedure_Code"].unique())[0],
        "cs_billed": 500.0, "cs_timely_limit": sorted(claims_readable["Payer_Timely_Filing_Limit_Days"].unique())[0],
        "cs_delay": 10, "cs_provider_vol": 50, "cs_prior_denials": 1,
        "cs_auth_req": False, "cs_auth_obt": True, "cs_eligibility": True, "cs_chronic": False,
        "cs_duplicate": False, "cs_bundling": False,
    }
    for k, v in _defaults.items():
        st.session_state.setdefault(k, v)

    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            form_type = st.selectbox("Claim form type", le["Claim_Form_Type"].classes_, key="cs_form_type")
            specialty = st.selectbox("Provider specialty", le["Provider_Specialty"].classes_, key="cs_specialty")
            payer = st.selectbox("Payer", le["Payer"].classes_, key="cs_payer")
            diagnosis = st.selectbox("Diagnosis code", le["Diagnosis_Code"].classes_, key="cs_diagnosis")
        with col2:
            modifier = st.selectbox("Modifier code", le["Modifier_Code"].classes_, key="cs_modifier")
            pos_code = st.selectbox("Place of service code", sorted(claims_readable["Place_of_Service_Code"].unique()), key="cs_pos_code")
            proc_code = st.selectbox("Procedure code", sorted(claims_readable["Procedure_Code"].unique()), key="cs_proc_code")
            billed = st.number_input("Billed amount ($)", 10.0, 25000.0, step=10.0, key="cs_billed")
        with col3:
            timely_limit = st.selectbox("Payer timely filing limit (days)", sorted(claims_readable["Payer_Timely_Filing_Limit_Days"].unique()), key="cs_timely_limit")
            delay = st.slider("Days between service and submission", 0, 300, key="cs_delay")
            provider_vol = st.slider("Provider monthly claim volume", 0, 300, key="cs_provider_vol")
            prior_denials = st.slider("Provider-payer prior denial count", 0, 10, key="cs_prior_denials")

        col4, col5, col6, col7 = st.columns(4)
        with col4: auth_req = st.checkbox("Prior auth required", key="cs_auth_req")
        with col5: auth_obt = st.checkbox("Prior auth obtained", key="cs_auth_obt")
        with col6: eligibility = st.checkbox("Eligibility verified", key="cs_eligibility")
        with col7: chronic = st.checkbox("Chronic condition on file", key="cs_chronic")
        col8, col9 = st.columns(2)
        with col8: duplicate = st.checkbox("Possible duplicate claim", key="cs_duplicate")
        with col9: bundling = st.checkbox("NCCI bundling flagged", key="cs_bundling")

        submitted = st.form_submit_button("\u26A1 Scrub this claim", type="primary", use_container_width=True)

    if submitted:
        claim_dict = {
            "Prior_Auth_Required": auth_req, "Prior_Auth_Obtained": auth_obt,
            "Eligibility_Verified": eligibility, "Duplicate_Claim_Flag": duplicate,
            "NCCI_Bundling_Flag": bundling, "Clearinghouse_Validation_Passed": True,
        }
        rule_edits = run_rule_edits(claim_dict)

        row = pd.DataFrame([{
            "Claim_Form_Type": le["Claim_Form_Type"].transform([form_type])[0],
            "Provider_Specialty": le["Provider_Specialty"].transform([specialty])[0],
            "Payer": le["Payer"].transform([payer])[0],
            "Place_of_Service_Code": pos_code,
            "Diagnosis_Code": le["Diagnosis_Code"].transform([diagnosis])[0],
            "Procedure_Code": proc_code,
            "Modifier_Code": le["Modifier_Code"].transform([modifier])[0],
            "Billed_Amount": billed,
            "Payer_Timely_Filing_Limit_Days": timely_limit,
            "Prior_Auth_Required": auth_req,
            "Prior_Auth_Obtained": auth_obt,
            "Eligibility_Verified": eligibility,
            "Duplicate_Claim_Flag": duplicate,
            "NCCI_Bundling_Flag": bundling,
            "Clearinghouse_Validation_Passed": True,
            "Provider_Monthly_Claim_Volume": provider_vol,
            "Provider_Payer_Prior_Denial_Count": prior_denials,
            "Chronic_Condition_Flag": chronic,
            "Service_Month": 6, "Submission_Month": 6,
            "Service_DayOfWeek": 2, "Submission_DayOfWeek": 4,
            "Claim_Submission_Delay": delay,
        }])[FEATURE_COLS]

        proba = float(artifacts["binary_model"].predict_proba(row)[0, 1])
        tier = risk_tier_from_proba(proba)

        result_col, gauge_col = st.columns([1.3, 1])

        with gauge_col:
            panel_open("Denial risk", "\U0001F4C8")
            st.markdown(f'<div style="text-align:center;">{risk_gauge_svg(proba, tier, PROBA_MIN, PROBA_MAX)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;">{risk_chip(tier)}</div>', unsafe_allow_html=True)
            if tier == "Low" and LOTTIE_AVAILABLE:
                lottie_json = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json")
                if lottie_json:
                    st_lottie(lottie_json, height=90, key="success_lottie")
            panel_close()

        with result_col:
            panel_open("Payer edit findings", "\U0001F50D")
            if not rule_edits:
                st.markdown(
                    '<div class="ledger-row clear"><span class="ledger-code">CLEAR</span>'
                    'No hard payer edits triggered.</div>', unsafe_allow_html=True
                )
            for severity, code, msg in rule_edits:
                st.markdown(
                    f'<div class="ledger-row {severity}"><span class="ledger-code">{code}</span>{msg}</div>',
                    unsafe_allow_html=True,
                )
            panel_close()

            carc_display_rows = []
            if proba > 0.429:  # aligned with Medium/Low tier boundary
                carc_row = row.copy()
                carc_proba = artifacts["carc_model"].predict_proba(carc_row)[0]
                top_idx = np.argsort(carc_proba)[::-1][:3]
                panel_open("Likely denial reason if denied", "\U0001F9EC")
                for idx in top_idx:
                    code = artifacts["carc_encoder"].classes_[idx]
                    desc = CARC_DESCRIPTIONS.get(code, "")
                    likelihood = f"{carc_proba[idx]:.0%}"
                    st.markdown(
                        f'<div class="ledger-row soft"><span class="ledger-code">{code}</span>'
                        f'{desc} \u2014 {likelihood} likelihood</div>',
                        unsafe_allow_html=True,
                    )
                    carc_display_rows.append([code, desc, likelihood])
                panel_close()

        panel_open("Why the model scored it this way (SHAP)", "\U0001F9E0")
        shap_vals = artifacts["explainer"].shap_values(row)[0]
        shap_df = pd.DataFrame({"Feature": FEATURE_COLS, "Impact": shap_vals})
        shap_df = shap_df.reindex(shap_df.Impact.abs().sort_values(ascending=False).index).head(8)
        fig = go.Figure(go.Bar(
            x=shap_df["Impact"], y=shap_df["Feature"], orientation="h",
            marker_color=["#FB4B4B" if v > 0 else "#22D3EE" for v in shap_df["Impact"]],
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                           xaxis_title="Impact on denial risk (right = increases risk)",
                           hovermode="y unified")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        panel_close()

        # ---- Downloadable PDF claim report ----
        edit_rows_for_pdf = [[sev.upper(), code, msg] for sev, code, msg in rule_edits]
        shap_rows_for_pdf = [[feat, f"{val:+.4f}"] for feat, val in zip(shap_df["Feature"], shap_df["Impact"])]
        pdf_bytes = generate_claim_pdf(f"{proba:.1%}", tier, edit_rows_for_pdf, shap_rows_for_pdf, carc_display_rows)
        st.download_button(
            "\U0001F4C4 Download claim report (PDF)", data=pdf_bytes,
            file_name="claim_scrubbing_report.pdf", mime="application/pdf",
            use_container_width=True,
        )

elif page == "Risk Queue":
    app_header("Pre-Submission Risk Queue", "held-out test set \u00b7 ranked by denial probability")

    X_test = artifacts["X_test"]
    proba = artifacts["binary_model"].predict_proba(X_test)[:, 1]
    queue = X_test.copy()
    queue["Denial_Probability"] = proba
    queue["Risk_Tier"] = [risk_tier_from_proba(p) for p in proba]
    queue = queue.sort_values("Denial_Probability", ascending=False)

    tier_filter = st.multiselect("Filter by risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    filtered = queue[queue["Risk_Tier"].isin(tier_filter)]

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Claims in queue", f"{len(filtered):,}")
    with c2: kpi_card("High risk", f"{(filtered.Risk_Tier=='High').sum():,}")
    with c3: kpi_card("Avg. risk score", f"{filtered.Denial_Probability.mean():.0%}" if len(filtered) else "\u2014")

    st.write("")
    panel_open("Queue", "\U0001F4CB")
    display_cols = ["Billed_Amount", "Provider_Payer_Prior_Denial_Count", "Prior_Auth_Obtained",
                     "Eligibility_Verified", "Denial_Probability", "Risk_Tier"]
    st.dataframe(
        filtered[display_cols].style.format({"Denial_Probability": "{:.1%}", "Billed_Amount": "${:.2f}"}),
        use_container_width=True, height=460,
    )
    excel_bytes = df_to_excel_bytes(filtered[display_cols], sheet_name="Risk Queue")
    st.download_button(
        "\U0001F4CA Export queue to Excel", data=excel_bytes,
        file_name="risk_queue.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    panel_close()

elif page == "Model Performance":
    app_header("Model Performance", "held-out validation, both models")

    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    X_test, y_test = artifacts["X_test"], artifacts["y_test"]
    y_prob = artifacts["binary_model"].predict_proba(X_test)[:, 1]
    y_pred = artifacts["binary_model"].predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    X_test_carc, y_test_carc = artifacts["X_test_carc"], artifacts["y_test_carc"]
    carc_pred = artifacts["carc_model"].predict(X_test_carc)
    carc_acc = accuracy_score(y_test_carc, carc_pred)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Model 1 \u00b7 Accuracy", f"{acc:.1%}", "denial prediction")
    with c2: kpi_card("Model 1 \u00b7 ROC-AUC", f"{auc:.3f}", "denial prediction")
    with c3: kpi_card("Model 2 \u00b7 Accuracy", f"{carc_acc:.1%}", "CARC reason, 8 classes")
    with c4: kpi_card("Bayes ceiling", "82.5%", "theoretical max, this dataset")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        panel_open("Model 1 \u2014 Feature importance (SHAP)", "\U0001F4CA")
        shap_vals = artifacts["explainer"].shap_values(X_test.sample(min(1500, len(X_test)), random_state=42))
        importance = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False).head(10)
        fig = px.bar(importance[::-1], orientation="h", labels={"value": "Mean |SHAP value|", "index": ""})
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, height=380)
        fig.update_traces(marker_color="#22D3EE")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        panel_close()

    with col2:
        panel_open("Model 2 \u2014 Per-class performance", "\U0001F9EC")
        report = classification_report(
            y_test_carc, carc_pred, target_names=artifacts["carc_encoder"].classes_, output_dict=True
        )
        report_df = pd.DataFrame(report).T.iloc[:-3][["precision", "recall", "f1-score", "support"]]
        st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}),
                     use_container_width=True)
        panel_close()
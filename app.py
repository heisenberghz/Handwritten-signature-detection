import sys
import time
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch

project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src.predict import SignaturePredictor
from src.verify_signature import SignatureVerifier
from config import IMAGE_SIZE, CHECKPOINT_DIR

st.set_page_config(
    page_title="Signature Forensics AI",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] > div {
    background-color: #1e293b;
    border-right: 1px solid #334155;
}

section[data-testid="stSidebar"] .stButton button {
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: #94a3b8;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 14px;
    transition: all 0.2s ease;
    margin-bottom: 2px;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.05);
    color: #e2e8f0;
    border-color: #334155;
}

section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(16,185,129,0.1);
    color: #10b981;
    border-color: rgba(16,185,129,0.3);
    font-weight: 600;
}

section[data-testid="stSidebar"] .stButton button[kind="primaryFormSubmit"] {
    background: rgba(16,185,129,0.1);
    color: #10b981;
    border-color: rgba(16,185,129,0.3);
    font-weight: 600;
}

.sidebar-brand {
    font-size: 18px;
    font-weight: 700;
    padding: 8px 4px;
    color: #e2e8f0;
    letter-spacing: -0.5px;
}

.sidebar-brand span.hl {
    color: #10b981;
}

.sidebar-footer {
    position: fixed;
    bottom: 16px;
    left: 16px;
    right: 16px;
    font-size: 11px;
    color: #475569;
}

.metric-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: #475569;
    transform: translateY(-1px);
}

.metric-value {
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    font-size: 32px;
    font-weight: 700;
    color: #e2e8f0;
}

.metric-label {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
    letter-spacing: 0.3px;
}

.cta-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    cursor: pointer;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.cta-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.cta-card.green {
    border-color: rgba(16,185,129,0.3);
}
.cta-card.green:hover {
    border-color: #10b981;
    box-shadow: 0 8px 30px rgba(16,185,129,0.15);
}
.cta-card.blue {
    border-color: rgba(59,130,246,0.3);
}
.cta-card.blue:hover {
    border-color: #3b82f6;
    box-shadow: 0 8px 30px rgba(59,130,246,0.15);
}

.cta-icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.cta-title {
    font-size: 20px;
    font-weight: 600;
    color: #e2e8f0;
}

.cta-desc {
    font-size: 13px;
    color: #64748b;
    margin-top: 6px;
}

.verdict-card {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}

.verdict-card.genuine {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.4);
    box-shadow: 0 0 30px rgba(16,185,129,0.1);
    animation: pulseGlow 2s ease-in-out infinite;
}

.verdict-card.forged {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    box-shadow: 0 0 30px rgba(239,68,68,0.1);
    animation: shakeAlert 0.5s ease-in-out;
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.1); }
    50% { box-shadow: 0 0 40px rgba(16,185,129,0.2); }
}

@keyframes shakeAlert {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-3px); }
    20%, 40%, 60%, 80% { transform: translateX(3px); }
}

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.badge.genuine {
    background: rgba(16,185,129,0.15);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
}

.badge.forged {
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.3);
}

.badge.match {
    background: rgba(16,185,129,0.15);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
}

.badge.mismatch {
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.3);
}

.prob-bar-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 4px;
}

.prob-track {
    height: 20px;
    background: #0f172a;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 12px;
}

.prob-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease-in-out;
}

.result-section {
    animation: fadeInUp 0.4s ease-out;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

div[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.6);
    border: 2px dashed #334155;
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #10b981;
    background: rgba(30, 41, 59, 0.8);
}

div[data-testid="stFileUploader"] section {
    padding: 0;
}

.st-emotion-cache-1aehpvj {
    color: #94a3b8;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 15px;
    padding: 10px 28px;
    transition: all 0.25s ease;
    border: none;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    border: none;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(16,185,129,0.3);
}

div.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
}

div.stButton > button[kind="secondary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(59,130,246,0.3);
}

.stats-row {
    display: flex;
    gap: 16px;
    margin: 24px 0;
}

.stats-card {
    flex: 1;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.stats-number {
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    font-size: 28px;
    font-weight: 700;
    color: #e2e8f0;
}

.stats-label {
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
}

h1, h2, h3 {
    color: #e2e8f0 !important;
}

p, li, .stMarkdown {
    color: #cbd5e1;
}

.stExpander {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    margin: 8px 0;
}

.stExpander > details > summary {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 500;
}

pre, code {
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
}

.hero-section {
    text-align: center;
    padding: 32px 0 16px 0;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    color: #64748b;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.5;
}

.hero-svg {
    margin-bottom: 16px;
}

.disclaimer-banner {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    color: #a78bfa;
    text-align: center;
    margin-top: 24px;
}

.legal-footer {
    font-size: 11px;
    color: #475569;
    text-align: center;
    padding: 16px;
    border-top: 1px solid #1e293b;
    margin-top: 40px;
}

.upload-hint {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    padding: 8px;
}

.device-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    border-radius: 6px;
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    font-size: 12px;
    color: #94a3b8;
}

[data-testid="stNotification"] {
    background: #1e293b;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ─────────────────────────────────────
for key in ["page", "temp_dir", "mode1_path", "mode1_result", "mode1_binary",
            "mode2_ref_path", "mode2_query_path", "mode2_result"]:
    if key not in st.session_state:
        if key == "page":
            st.session_state[key] = "Home"
        elif key == "temp_dir":
            st.session_state[key] = Path(tempfile.mkdtemp(prefix="sigfx_"))
        else:
            st.session_state[key] = None

# ─── Helper Functions ────────────────────────────────────────

def save_upload(uploaded_file, tag: str) -> Path:
    ext = Path(uploaded_file.name).suffix or ".png"
    path = st.session_state.temp_dir / f"{tag}_{uuid.uuid4().hex}{ext}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def preprocess_signature(image_path: Path):
    """Matching predict.py pipeline exactly. Returns (tensor, binary_uint8_display, original_gray)."""
    try:
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            return None, None, None

        img = cv2.GaussianBlur(original, (5, 5), 0)

        mean_val = np.mean(img)
        if mean_val > 127:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = binary[y:y + h, x:x + w]
            canvas = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            start_y = max(0, (IMAGE_SIZE[0] - h) // 2)
            start_x = max(0, (IMAGE_SIZE[1] - w) // 2)
            end_y = min(start_y + h, IMAGE_SIZE[0])
            end_x = min(start_x + w, IMAGE_SIZE[1])
            canvas[start_y:end_y, start_x:end_x] = cropped[:end_y - start_y, :end_x - start_x]
            binary = canvas

        binary_display = binary.copy()
        tensor = torch.from_numpy(binary.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        return tensor, binary_display, original
    except Exception:
        return None, None, None

@st.cache_resource
def get_predictor():
    from config import MODEL_FILENAME
    model_file = CHECKPOINT_DIR / MODEL_FILENAME
    if not model_file.exists():
        st.error(f"Model not found at: {model_file}")
        st.stop()
    return SignaturePredictor(model_file)

@st.cache_resource
def get_verifier():
    from config import MODEL_FILENAME
    model_file = CHECKPOINT_DIR / MODEL_FILENAME
    if not model_file.exists():
        st.error(f"Model not found at: {model_file}")
        st.stop()
    return SignatureVerifier(model_file)

def circular_progress_svg(percentage: float, color: str = "#10b981", size: int = 140):
    cx = cy = size // 2
    r = int(size * 0.45)
    circumference = 2 * 3.14159 * r
    offset = circumference * (1 - min(percentage, 100) / 100)
    font_size = size // 5
    label_size = max(10, size // 12)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1e293b" stroke-width="{size//18}"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{size//18}"
            stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
            transform="rotate(-90, {cx}, {cy})" stroke-linecap="round"
            style="transition: stroke-dashoffset 0.8s ease-in-out;"/>
        <text x="{cx}" y="{cy - size//28}" text-anchor="middle" dominant-baseline="central"
            fill="#e2e8f0" font-family="'JetBrains Mono','SF Mono',monospace" font-size="{font_size}" font-weight="bold">
            {percentage:.1f}%
        </text>
        <text x="{cx}" y="{cy + size//8}" text-anchor="middle" dominant-baseline="central"
            fill="#64748b" font-family="sans-serif" font-size="{label_size}">
            Confidence
        </text>
    </svg>
    """

def make_gauge(score: float, threshold: float = None):
    if threshold is None:
        from config import SIMILARITY_THRESHOLD
        threshold = SIMILARITY_THRESHOLD
    color = "#ef4444"
    if score >= threshold:
        color = "#10b981"
    elif score >= 0.5:
        color = "#f59e0b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Similarity Score", "font": {"size": 16, "color": "#94a3b8"}},
        number={
            "font": {"size": 44, "color": "#e2e8f0", "family": "JetBrains Mono, SF Mono, monospace"},
            "valueformat": ".3f",
        },
        gauge={
            "axis": {
                "range": [0, 1],
                "tickwidth": 1,
                "tickcolor": "#64748b",
                "tickfont": {"size": 11, "color": "#94a3b8"},
            },
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 0.5], "color": "rgba(239,68,68,0.15)"},
                {"range": [0.5, 0.7], "color": "rgba(245,158,11,0.12)"},
                {"range": [0.7, 1.0], "color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#f59e0b", "width": 4},
                "thickness": 0.75,
                "value": threshold,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        height=320,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
    )
    return fig

def device_badge():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"<span class='device-badge'>🖥 GPU: {name}</span>"
    return "<span class='device-badge'>💻 CPU</span>"

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🖋️ <span class="hl">Sig</span>Forensics</div>', unsafe_allow_html=True)
    st.markdown("---")

    current = st.session_state.page

    nav_items = [
        ("Home", "🏠 Home"),
        ("Mode1", "🔍 Single Analysis"),
        ("Mode2", "⚖️ Pairwise Verification"),
    ]

    for pid, label in nav_items:
        kind = "primary" if current == pid else "secondary"
        if st.button(label, width='stretch', key=f"nav_{pid}", type=kind):
            st.session_state.page = pid
            st.rerun()

    st.markdown("---")
    st.markdown(device_badge(), unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-top:40px;font-size:11px;color:#475569;'>"
        "Educational ML Demo<br>Not certified for legal proceedings"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Pages ───────────────────────────────────────────────────

def home_page():
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-svg">
        <svg width="180" height="60" viewBox="0 0 180 60">
            <path d="M10,45 Q30,10 50,35 T90,25 T130,40 T170,20"
                  fill="none" stroke="#10b981" stroke-width="2.5"
                  stroke-linecap="round" stroke-linejoin="round"
                  stroke-dasharray="300" stroke-dashoffset="0">
                <animate attributeName="stroke-dashoffset"
                         from="300" to="0" dur="1.5s" fill="freeze"/>
            </path>
        </svg>
    </div>
    <div class="hero-title">Signature Forensics AI</div>
    <div class="hero-subtitle">
        Deep learning–based forensic analysis for handwritten signature verification.
        Two complementary workflows for forgery detection.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="cta-card green">
            <div class="cta-icon">🔍</div>
            <div class="cta-title">Single Signature Analysis</div>
            <div class="cta-desc">Upload one signature and get a genuine/forged verdict with confidence metrics</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze a Signature", key="cta_mode1", width='stretch', type="primary"):
            st.session_state.page = "Mode1"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="cta-card blue">
            <div class="cta-icon">⚖️</div>
            <div class="cta-title">Verify Against Reference</div>
            <div class="cta-desc">Compare a questioned signature against a known genuine reference using deep features</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Verification", key="cta_mode2", width='stretch', type="secondary"):
            st.session_state.page = "Mode2"
            st.rerun()

    st.markdown("""
    <div class="stats-row">
        <div class="stats-card">
            <div class="stats-number">76.39%</div>
            <div class="stats-label">Test Accuracy</div>
        </div>
        <div class="stats-card">
            <div class="stats-number">87.96%</div>
            <div class="stats-label">Recall (Genuine)</div>
        </div>
        <div class="stats-card">
            <div class="stats-number">87.78%</div>
            <div class="stats-label">ROC-AUC</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Model Performance & Baseline Comparison"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**CNN v2 Metrics**")
            st.markdown("""
| Metric | Value |
|--------|-------|
| Test Accuracy | 76.39% |
| Precision | 71.43% |
| Recall | 87.96% |
| F1-Score | 78.84% |
| ROC-AUC | 87.78% |
""")
        with col_b:
            st.markdown("**Baseline Comparison**")
            st.markdown("""
| Model | Accuracy |
|-------|----------|
| SVM | 61.34% |
| Random Forest | 63.19% |
| CNN v1 (overfit) | 61.11% |
| CNN v2 (production) | **76.39%** |
""")

    st.markdown("""
    <div class="legal-footer">
        🛡️ Educational ML Demo — Not Certified for Legal Proceedings.
        Trained on the CEDAR signature dataset.
    </div>
    """, unsafe_allow_html=True)


def mode1_page():
    st.markdown("## 🔍 Single Signature Analysis")
    st.markdown("Upload a signature image to classify it as **Genuine** or **Forged**.")

    uploaded = st.file_uploader(
        "Drop signature image here or click to browse",
        type=["png", "jpg", "jpeg"],
        key="mode1_uploader",
        help="PNG, JPG, JPEG — max 10MB",
    )

    col_preview, _ = st.columns([1, 1])
    with col_preview:
        if uploaded is not None and (st.session_state.mode1_path is None or uploaded.name != st.session_state.get("mode1_name")):
            with st.spinner("Processing image..."):
                st.session_state.mode1_path = save_upload(uploaded, "m1")
                st.session_state.mode1_name = uploaded.name
                st.session_state.mode1_result = None
                st.session_state.mode1_binary = None

        if st.session_state.mode1_path and st.session_state.mode1_path.exists():
            file_size = st.session_state.mode1_path.stat().st_size
            orig = cv2.imread(str(st.session_state.mode1_path), cv2.IMREAD_GRAYSCALE)
            if orig is not None:
                st.image(orig, caption=f"Uploaded: {st.session_state.mode1_name} ({orig.shape[1]}×{orig.shape[0]}, {file_size//1024}KB)", width='stretch', channels="L")

    analyze_col = st.columns([1, 2, 1])[1]
    with analyze_col:
        btn_disabled = st.session_state.mode1_path is None
        if st.button("🧬 Analyze Signature", width='stretch', type="primary", disabled=btn_disabled):
            if st.session_state.mode1_path is None:
                st.warning("Please upload an image first.")
                st.stop()

            with st.spinner("Analyzing with CNN..."):
                tensor, binary_display, original_display = preprocess_signature(st.session_state.mode1_path)
                if tensor is None:
                    st.error("Could not process image. Please upload a valid PNG/JPG with a visible signature.")
                    st.stop()

                st.session_state.mode1_binary = binary_display

                pred = get_predictor()
                t0 = time.perf_counter()
                result = pred.predict(st.session_state.mode1_path)
                dt = (time.perf_counter() - t0) * 1000

                result["inference_time_ms"] = dt
                st.session_state.mode1_result = result

    result = st.session_state.mode1_result
    if result is not None:
        st.markdown("---")
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        is_genuine = result["prediction"] == "Genuine"
        color = "#10b981" if is_genuine else "#ef4444"
        verdict_class = "genuine" if is_genuine else "forged"
        badge_text = "✅ AUTHENTIC" if is_genuine else "❌ SUSPICIOUS"
        badge_class = "genuine" if is_genuine else "forged"

        col_viz, col_verdict = st.columns([1, 1])

        with col_viz:
            st.markdown("**🖼 How the AI sees your signature**")
            binary_disp = st.session_state.mode1_binary
            if binary_disp is not None:
                before, after = st.columns(2)
                with before:
                    orig_rgb = cv2.imread(str(st.session_state.mode1_path))
                    if orig_rgb is not None:
                        st.image(cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2RGB), caption="Original", width='stretch')
                with after:
                    st.image(binary_disp, caption="Preprocessed (256×256 binary)", width='stretch', channels="L")

        with col_verdict:
            st.markdown(f"""
            <div class="verdict-card {verdict_class}" style="text-align:center;">
                <div style="margin-bottom:12px;">
                    <span class="badge {badge_class}" style="font-size:18px;padding:8px 24px;">{badge_text}</span>
                </div>
                <div style="display:flex;justify-content:center;margin:12px 0;">
                    {circular_progress_svg(result['confidence'] * 100, color)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**📊 Probability Breakdown**")
        genuine_pct = result["genuine_probability"] * 100
        forged_pct = result["forged_probability"] * 100
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-row">
                <span>Genuine</span>
                <span>{genuine_pct:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{genuine_pct}%;background:linear-gradient(90deg,#10b981,#059669);"></div>
            </div>
            <div class="prob-row">
                <span>Forged</span>
                <span>{forged_pct:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{forged_pct}%;background:linear-gradient(90deg,#ef4444,#dc2626);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Technical Readout"):
            st.markdown(f"""
| Property | Value |
|----------|-------|
| Model | CNN v2 (~8.5M parameters) |
| Inference Time | {result.get('inference_time_ms', 0):.0f}ms |
| Device | {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} |
| Preprocessing | Grayscale → GaussianBlur(5×5) → Otsu → MorphOpen(2×2) → CenterPad(256×256) → Normalize |
| Input Shape | [1, 1, 256, 256] |
            """)

        st.markdown("""
        <div class="disclaimer-banner">
            ⚠️ Trained on CEDAR dataset. Performance may vary on out-of-distribution samples
            (different pens, paper, scanning quality).
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def mode2_page():
    st.markdown("## ⚖️ Verify Against Reference")
    st.markdown("Upload a **known genuine reference** and a **questioned signature** to compare.")

    col_ref, col_query = st.columns(2)

    with col_ref:
        st.markdown("**📎 Reference (Known Genuine)**")
        ref_file = st.file_uploader("Upload reference signature", type=["png", "jpg", "jpeg"], key="ref_uploader", label_visibility="collapsed")
        if ref_file is not None and (st.session_state.mode2_ref_path is None or ref_file.name != st.session_state.get("mode2_ref_name")):
            with st.spinner("Processing reference..."):
                st.session_state.mode2_ref_path = save_upload(ref_file, "ref")
                st.session_state.mode2_ref_name = ref_file.name
                st.session_state.mode2_result = None

        if st.session_state.mode2_ref_path and st.session_state.mode2_ref_path.exists():
            ref_img = cv2.imread(str(st.session_state.mode2_ref_path), cv2.IMREAD_GRAYSCALE)
            if ref_img is not None:
                st.image(ref_img, caption=st.session_state.mode2_ref_name, width='stretch', channels="L")

    with col_query:
        st.markdown("**📎 Questioned Signature**")
        q_file = st.file_uploader("Upload questioned signature", type=["png", "jpg", "jpeg"], key="query_uploader", label_visibility="collapsed")
        if q_file is not None and (st.session_state.mode2_query_path is None or q_file.name != st.session_state.get("mode2_query_name")):
            with st.spinner("Processing questioned..."):
                st.session_state.mode2_query_path = save_upload(q_file, "query")
                st.session_state.mode2_query_name = q_file.name
                st.session_state.mode2_result = None

        if st.session_state.mode2_query_path and st.session_state.mode2_query_path.exists():
            q_img = cv2.imread(str(st.session_state.mode2_query_path), cv2.IMREAD_GRAYSCALE)
            if q_img is not None:
                st.image(q_img, caption=st.session_state.mode2_query_name, width='stretch', channels="L")

    verify_col = st.columns([1, 2, 1])[1]
    with verify_col:
        can_verify = (
            st.session_state.mode2_ref_path is not None
            and st.session_state.mode2_ref_path.exists()
            and st.session_state.mode2_query_path is not None
            and st.session_state.mode2_query_path.exists()
        )
        if st.button("🔗 Verify Match", width='stretch', type="secondary", disabled=not can_verify):
            if not can_verify:
                st.warning("Please upload both reference and questioned signatures.")
                st.stop()

            with st.spinner("Computing deep feature similarity..."):
                verifier = get_verifier()
                t0 = time.perf_counter()
                result = verifier.verify(st.session_state.mode2_ref_path, st.session_state.mode2_query_path)
                dt = (time.perf_counter() - t0) * 1000
                result["inference_time_ms"] = dt
                st.session_state.mode2_result = result

    result = st.session_state.mode2_result
    if result is not None:
        st.markdown("---")
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        score = result["similarity_score"]
        is_match = result["is_match"]
        confidence = result["confidence"]
        threshold = result["threshold"]

        score_color = "#ef4444"
        if score >= threshold:
            score_color = "#10b981"
        elif score >= 0.5:
            score_color = "#f59e0b"

        verdict_text = "✅ MATCH — Signatures are consistent" if is_match else "❌ MISMATCH — Signatures differ significantly"
        verdict_sub = (
            "The questioned signature exhibits similar deep spatial features to the reference."
            if is_match
            else "The questioned signature shows divergent stroke patterns and texture characteristics."
        )

        st.markdown(f"""
        <div style="text-align:center;margin-bottom:16px;">
            <span style="font-family:'JetBrains Mono','SF Mono',monospace;font-size:48px;font-weight:700;color:{score_color};">
                {score:.3f}
            </span>
            <div style="color:#64748b;font-size:14px;">Similarity Score (0.000 – 1.000)</div>
        </div>
        """, unsafe_allow_html=True)

        gauge = make_gauge(score, threshold)
        st.plotly_chart(gauge, width='stretch', config={"displayModeBar": False})

        st.markdown(f"""
        <div class="verdict-card {'genuine' if is_match else 'forged'}" style="text-align:center;margin-top:16px;">
            <span class="badge {'match' if is_match else 'mismatch'}" style="font-size:16px;padding:8px 24px;">{verdict_text}</span>
            <p style="color:#94a3b8;font-size:14px;margin-top:12px;">{verdict_sub}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center;margin-top:16px;">
            <span style="font-family:'JetBrains Mono','SF Mono',monospace;font-size:36px;font-weight:700;color:#e2e8f0;">
                {confidence:.1%}
            </span>
            <div style="color:#64748b;font-size:13px;">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Technical Readout"):
            st.markdown(f"""
| Property | Value |
|----------|-------|
| Feature Vector Dimension | 32,768 |
| Similarity Metric | Cosine Similarity: cos(θ) = (A·B) / (‖A‖·‖B‖) |
| Match Threshold | {threshold} |
| Device | {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} |
| Inference Time | {result.get('inference_time_ms', 0):.0f}ms |
| Preprocessing | Grayscale → GaussianBlur(5×5) → Otsu → MorphOpen(2×2) → CenterPad(256×256) → Normalize |
            """)

        st.markdown('</div>', unsafe_allow_html=True)


# ─── Routing ─────────────────────────────────────────────────
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Mode1":
    mode1_page()
elif st.session_state.page == "Mode2":
    mode2_page()

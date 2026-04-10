import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from openai import OpenAI
import os
import gdown

# ============================================================
#  API SETUP  (unchanged)
# ============================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_remedy(disease_name):
    try:
        prompt = f"""
        You are an agricultural expert.

        Provide:
        1. Disease explanation
        2. Treatment steps
        3. Prevention tips

        Disease: {disease_name}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except:
        return f"""
AI service unavailable.

Basic Remedy for {disease_name}:
- Remove infected leaves
- Apply suitable fungicide
- Avoid overwatering
- Maintain proper spacing
"""

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NeuralLeaf – Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  GLOBAL CSS  — Light Green Apple-Like Premium Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Reset & Base ────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden; display: none; }

/* ── Page background — soft sage gradient ────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #f0faf4 0%, #e6f7ed 40%, #f5fdf7 100%) !important;
    min-height: 100vh;
}

/* Subtle organic texture overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle at 15% 20%, rgba(134, 239, 172, 0.18) 0%, transparent 55%),
        radial-gradient(circle at 85% 75%, rgba(74, 222, 128, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(240, 253, 244, 0.3) 0%, transparent 80%);
    pointer-events: none;
    z-index: 0;
}

/* ── Block container ─────────────────────────────── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 1080px !important;
    position: relative;
    z-index: 1;
}

/* ── Design Tokens ───────────────────────────────── */
:root {
    --green-50:   #f0fdf4;
    --green-100:  #dcfce7;
    --green-200:  #bbf7d0;
    --green-400:  #4ade80;
    --green-500:  #22c55e;
    --green-600:  #16a34a;
    --green-700:  #15803d;
    --sage-50:    #f8faf8;
    --sage-100:   #eef5ee;
    --sage-200:   #d4e8d4;
    --amber-500:  #f59e0b;
    --red-400:    #f87171;
    --blue-400:   #60a5fa;
    --text-primary:   #1a2e1a;
    --text-secondary: #3d5a3d;
    --text-muted:     #6b8f6b;
    --text-light:     #9ab89a;
    --white-glass:    rgba(255, 255, 255, 0.72);
    --white-glass-hv: rgba(255, 255, 255, 0.88);
    --border-light:   rgba(134, 239, 172, 0.35);
    --border-mid:     rgba(74, 222, 128, 0.25);
    --shadow-sm:  0 1px 4px rgba(22, 101, 52, 0.06), 0 2px 12px rgba(22, 101, 52, 0.04);
    --shadow-md:  0 4px 20px rgba(22, 101, 52, 0.08), 0 1px 4px rgba(22, 101, 52, 0.05);
    --shadow-lg:  0 8px 40px rgba(22, 101, 52, 0.10), 0 2px 8px rgba(22, 101, 52, 0.06);
    --radius-xl:  20px;
    --radius-lg:  14px;
    --radius-md:  10px;
    --radius-sm:  7px;
}

/* ── Typography ──────────────────────────────────── */
h1, h2, h3 { font-family: 'DM Serif Display', Georgia, serif !important; }
p, li, span, div { font-family: 'DM Sans', sans-serif !important; }

/* ── Hero ────────────────────────────────────────── */
.hero-wrap {
    text-align: center;
    padding: 3.2rem 1.5rem 2.4rem;
    position: relative;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(74, 222, 128, 0.12);
    border: 1px solid rgba(74, 222, 128, 0.3);
    color: var(--green-600);
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 1.4rem;
}
.hero-logo-ring {
    width: 82px;
    height: 82px;
    background: linear-gradient(145deg, #ffffff, #e8faf0);
    border: 1.5px solid rgba(74, 222, 128, 0.4);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.4rem;
    margin: 0 auto 1.2rem;
    box-shadow: 0 4px 24px rgba(74, 222, 128, 0.2), 0 0 0 6px rgba(74, 222, 128, 0.06);
}
.hero-title {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 3.4rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1.12;
    margin: 0 0 0.6rem;
    letter-spacing: -0.5px;
}
.hero-title em {
    font-style: italic;
    color: var(--green-600);
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto 1.8rem;
    line-height: 1.65;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}
.badge {
    background: var(--white-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-light);
    color: var(--text-secondary);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.76rem;
    font-weight: 500;
    box-shadow: var(--shadow-sm);
}

/* ── Section Divider ─────────────────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,222,128,0.4), transparent);
    margin: 0.4rem 0 2rem;
    border: none;
}

/* ── Section Label ───────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--green-600);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Glass Card ──────────────────────────────────── */
.card {
    background: var(--white-glass);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-xl);
    padding: 1.6rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 0;
}
.card-sm {
    background: var(--white-glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem;
    box-shadow: var(--shadow-sm);
}

/* ── Upload section ──────────────────────────────── */
.upload-wrapper {
    background: linear-gradient(135deg, rgba(255,255,255,0.85) 0%, rgba(240,253,244,0.9) 100%);
    backdrop-filter: blur(20px);
    border: 2px dashed rgba(74, 222, 128, 0.45);
    border-radius: var(--radius-xl);
    padding: 2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
    box-shadow: var(--shadow-sm);
}
.upload-icon {
    font-size: 2.8rem;
    margin-bottom: 0.5rem;
    display: block;
}
.upload-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.upload-hint {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
}
.upload-formats {
    display: inline-flex;
    gap: 6px;
    margin-top: 0.8rem;
    flex-wrap: wrap;
    justify-content: center;
}
.fmt-chip {
    background: rgba(74, 222, 128, 0.1);
    border: 1px solid rgba(74, 222, 128, 0.25);
    color: var(--green-600);
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 4px;
    letter-spacing: 0.5px;
}

/* ── Streamlit file uploader inside upload-wrapper ── */
.upload-wrapper [data-testid="stFileUploader"] {
    background: transparent !important;
}
.upload-wrapper [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.6) !important;
    border: 1.5px solid rgba(74, 222, 128, 0.35) !important;
    border-radius: var(--radius-lg) !important;
}

/* ── Result Disease Chip ─────────────────────────── */
.result-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
    color: #fff;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 10px 22px;
    border-radius: 999px;
    box-shadow: 0 4px 16px rgba(22, 163, 74, 0.3);
    letter-spacing: 0.1px;
    margin-bottom: 0.9rem;
}
.result-pill.diseased {
    background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    box-shadow: 0 4px 16px rgba(217, 119, 6, 0.3);
}

/* ── Confidence Bar ──────────────────────────────── */
.conf-wrap {
    margin-top: 1rem;
}
.conf-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
}
.conf-lbl { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }
.conf-val { font-size: 0.9rem; font-weight: 700; color: var(--green-600); }
.conf-track {
    height: 7px;
    background: rgba(134, 239, 172, 0.25);
    border-radius: 999px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    transition: width 0.8s cubic-bezier(0.23, 1, 0.32, 1);
}

/* ── Metric pill row ─────────────────────────────── */
.metric-row {
    display: flex;
    gap: 10px;
    margin-top: 1.1rem;
}
.metric-pill {
    flex: 1;
    background: rgba(240, 253, 244, 0.8);
    border: 1px solid rgba(134, 239, 172, 0.4);
    border-radius: var(--radius-md);
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.metric-pill .mp-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    display: block;
}
.metric-pill .mp-lbl {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    display: block;
    margin-top: 2px;
}

/* ── Status strip ────────────────────────────────── */
.status-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: var(--radius-sm);
    font-size: 0.86rem;
    font-weight: 500;
    margin-bottom: 0.9rem;
}
.status-strip.healthy {
    background: rgba(74, 222, 128, 0.12);
    border: 1px solid rgba(74, 222, 128, 0.3);
    color: var(--green-700);
}
.status-strip.diseased {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #92400e;
}

/* ── Remedy Card ─────────────────────────────────── */
.remedy-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(240,253,244,0.95) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(134, 239, 172, 0.35);
    border-left: 4px solid var(--green-500);
    border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
    padding: 1.5rem 1.6rem;
    font-size: 0.94rem;
    line-height: 1.75;
    color: var(--text-secondary);
    box-shadow: var(--shadow-sm);
}

/* ── Tab overrides ───────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.6) !important;
    border-radius: var(--radius-lg) !important;
    padding: 4px !important;
    border: 1px solid var(--border-light) !important;
    box-shadow: var(--shadow-sm) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: var(--radius-md) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 6px 16px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: var(--green-700) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Streamlit metric override ───────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.8rem 1rem !important;
}

/* ── Download button ─────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-lg) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(22, 163, 74, 0.25) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    box-shadow: 0 6px 24px rgba(22, 163, 74, 0.38) !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ─────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--green-600) !important;
}

/* ── Info / success / warning boxes ─────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-lg) !important;
    border: none !important;
    font-size: 0.88rem !important;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fef9 0%, #f0faf4 100%) !important;
    border-right: 1px solid rgba(134, 239, 172, 0.35) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

.sb-brand {
    text-align: center;
    padding: 0 1rem 1.2rem;
    border-bottom: 1px solid rgba(134, 239, 172, 0.3);
    margin-bottom: 1.2rem;
}
.sb-logo-ring {
    width: 52px;
    height: 52px;
    background: linear-gradient(145deg, #ffffff, #e8faf0);
    border: 1.5px solid rgba(74, 222, 128, 0.4);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin: 0 auto 0.6rem;
    box-shadow: 0 2px 12px rgba(74, 222, 128, 0.18);
}
.sb-app-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: var(--text-primary);
    margin: 0;
}
.sb-tagline {
    font-size: 0.73rem;
    color: var(--text-muted);
    margin: 0;
}

.sb-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--green-600);
    padding: 0 0.2rem;
    margin: 1.1rem 0 0.5rem;
}
.sb-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.84rem;
    color: var(--text-secondary);
    padding: 5px 0.2rem;
    border-bottom: 1px solid rgba(134, 239, 172, 0.18);
    line-height: 1.4;
}
.sb-item:last-child { border-bottom: none; }
.sb-item .icon { font-size: 0.95rem; flex-shrink: 0; }

/* ── Image preview card ──────────────────────────── */
.img-meta {
    text-align: center;
    color: var(--text-light);
    font-size: 0.78rem;
    margin-top: 7px;
    font-weight: 400;
}

/* ── Card inner title ────────────────────────────── */
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--text-primary);
    margin: 0 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Step header ─────────────────────────────────── */
.step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.8rem;
}
.step-num {
    width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: #fff;
    border-radius: 50%;
    font-size: 0.73rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(22, 163, 74, 0.3);
}
.step-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--green-600);
}

/* ── Footer ──────────────────────────────────────── */
.footer-wrap {
    text-align: center;
    padding: 2rem 0 0.5rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(134, 239, 172, 0.3);
}
.footer-logo { font-size: 1.4rem; margin-bottom: 0.4rem; }
.footer-text {
    font-size: 0.8rem;
    color: var(--text-light);
    line-height: 1.8;
}
.footer-text strong { color: var(--green-600); font-weight: 600; }

/* ── Remove stray Streamlit vertical gaps ────────── */
.element-container:empty { display: none !important; }
div[data-testid="column"] > div:empty { display: none !important; }
.stMarkdown p { margin: 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo-ring">🌿</div>
        <p class="sb-app-name">NeuralLeaf</p>
        <p class="sb-tagline">Plant Disease AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item"><span class="icon">🎓</span> Final Year Project</div>
    <div class="sb-item"><span class="icon">🤖</span> Deep Learning + GPT-4o</div>
    <div class="sb-item"><span class="icon">🌾</span> Agriculture AI Domain</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item"><span class="icon">🧠</span> Architecture: ResNet-18</div>
    <div class="sb-item"><span class="icon">🏷️</span> Classes: 15 Disease Labels</div>
    <div class="sb-item"><span class="icon">🖼️</span> Input: 224 × 224 px</div>
    <div class="sb-item"><span class="icon">📦</span> Dataset: PlantVillage</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">Supported Plants</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item"><span class="icon">🫑</span> Pepper Bell</div>
    <div class="sb-item"><span class="icon">🥔</span> Potato</div>
    <div class="sb-item"><span class="icon">🍅</span> Tomato</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item"><span class="icon">1️⃣</span> Upload a clear leaf photo</div>
    <div class="sb-item"><span class="icon">2️⃣</span> Wait for AI prediction</div>
    <div class="sb-item"><span class="icon">3️⃣</span> Read remedy & prevention</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 For best results, use a well-lit photo of a single leaf against a plain background.", icon=None)


# ============================================================
#  HERO SECTION
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">🌱 AI-Powered Agriculture</div>
    <div class="hero-logo-ring">🌿</div>
    <h1 class="hero-title">Neural<em>Leaf</em></h1>
    <p class="hero-sub">
        Upload a plant leaf photo and get an instant AI-powered disease diagnosis
        with expert treatment recommendations.
    </p>
    <div class="badge-row">
        <span class="badge">🧠 ResNet-18</span>
        <span class="badge">⚡ GPT-4o Remedy</span>
        <span class="badge">🌾 PlantVillage Dataset</span>
        <span class="badge">🏷️ 15 Disease Classes</span>
    </div>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)


# ============================================================
#  MODEL DOWNLOAD  (unchanged)
# ============================================================
MODEL_ID   = "1llTeglJuXxsudW_1N11_SMXtgfTIZLXA"
MODEL_PATH = "plant_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Downloading model weights — first run only…"):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)


# ============================================================
#  LOAD MODEL  (unchanged)
# ============================================================
@st.cache_resource
def load_model():
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 15)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    m.eval()
    return m

model = load_model()

# ============================================================
#  TRANSFORM & CLASSES  (unchanged)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = [
    "Pepper Bell Healthy", "Potato Early Blight", "Potato Late Blight",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus",
    "Tomato Healthy", "Pepper Bell Bacterial Spot", "Potato Healthy"
]

def is_healthy(label: str) -> bool:
    return "healthy" in label.lower()


# ============================================================
#  STEP 1 — UPLOAD
# ============================================================
st.markdown("""
<div class="step-header">
    <div class="step-num">1</div>
    <div class="step-title">Upload Leaf Image</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-wrapper">
    <span class="upload-icon">🍃</span>
    <div class="upload-title">Drop your leaf photo here</div>
    <div class="upload-hint">Drag & drop or click below to browse</div>
    <div class="upload-formats">
        <span class="fmt-chip">JPG</span>
        <span class="fmt-chip">PNG</span>
        <span class="fmt-chip">JPEG</span>
        <span class="fmt-chip">WEBP</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Choose a leaf image",
    type=["jpg", "png", "jpeg", "webp"],
    help="Supported: JPG, PNG, JPEG, WEBP",
    label_visibility="collapsed"
)


# ============================================================
#  STEP 2 & 3 — RESULTS
# ============================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    # ---- Inference (unchanged logic) ----
    with torch.no_grad():
        outputs    = model(img_tensor)
        prob       = torch.softmax(outputs, dim=1)
        predicted  = torch.argmax(prob, dim=1)
        confidence = prob[0][predicted.item()].item() * 100

    label   = classes[predicted.item()]
    healthy = is_healthy(label)
    status_icon  = "✅" if healthy else "⚠️"
    pill_class   = "result-pill" if healthy else "result-pill diseased"
    strip_class  = "status-strip healthy" if healthy else "status-strip diseased"
    status_msg   = "Plant appears healthy — no disease detected." if healthy else "Disease detected — expert remedy provided below."

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step 2 header ──────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <div class="step-num">2</div>
        <div class="step-title">Diagnosis Results</div>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_res = st.columns([1, 1.2], gap="medium")

    # ── Left: image preview ────────────────────────────────
    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🍃 Uploaded Leaf</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f'<p class="img-meta">{uploaded_file.name} &nbsp;·&nbsp; {image.width} × {image.height} px</p>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right: prediction ──────────────────────────────────
    with col_res:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔬 AI Prediction</div>', unsafe_allow_html=True)

        # Disease pill
        st.markdown(
            f'<div class="{pill_class}">{status_icon} &nbsp;{label}</div>',
            unsafe_allow_html=True
        )

        # Status strip
        st.markdown(
            f'<div class="{strip_class}">{status_msg}</div>',
            unsafe_allow_html=True
        )

        # Confidence bar (pure HTML — no st.progress)
        fill_pct = int(confidence)
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-header">
                <span class="conf-lbl">Model Confidence</span>
                <span class="conf-val">{confidence:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{fill_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metric pills
        status_label = "Healthy ✅" if healthy else "Diseased ⚠️"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill">
                <span class="mp-val">{confidence:.1f}%</span>
                <span class="mp-lbl">Confidence</span>
            </div>
            <div class="metric-pill">
                <span class="mp-val">{'✅' if healthy else '⚠️'}</span>
                <span class="mp-lbl">Status</span>
            </div>
            <div class="metric-pill">
                <span class="mp-val">15</span>
                <span class="mp-lbl">Classes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 3: Remedy ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-header">
        <div class="step-num">3</div>
        <div class="step-title">AI Remedy & Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🤖 Generating expert remedy…"):
        remedy = generate_remedy(label)

    tab_remedy, tab_tips, tab_raw = st.tabs(["📋 Full Remedy", "📌 Quick Tips", "🗒️ Raw Output"])

    with tab_remedy:
        st.markdown(
            f'<div class="remedy-card">{remedy.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

    with tab_tips:
        st.markdown('<div class="card-sm">', unsafe_allow_html=True)
        if healthy:
            st.success("Your plant looks healthy! Here are general tips to keep it that way:")
            st.markdown("""
- ✅ Water consistently but avoid waterlogging
- ✅ Ensure good sunlight and air circulation
- ✅ Inspect leaves weekly for early signs of disease
- ✅ Use balanced fertiliser during the growing season
- ✅ Rotate crops seasonally to prevent soil-borne disease
            """)
        else:
            st.warning(f"Quick action checklist for **{label}**:")
            st.markdown("""
- 🔴 Isolate the affected plant immediately
- 🔴 Remove and destroy visibly infected leaves
- 🟡 Apply appropriate fungicide / pesticide
- 🟡 Reduce overhead watering; water at the base
- 🟢 Monitor surrounding plants for spread
- 🟢 Consult a local agricultural officer if severe
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_raw:
        st.markdown('<div class="card-sm">', unsafe_allow_html=True)
        st.code(remedy, language="markdown")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️  Download Remedy Report",
        data=f"Disease: {label}\nConfidence: {confidence:.1f}%\n\n{remedy}",
        file_name=f"NeuralLeaf_{label.replace(' ', '_')}_remedy.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
<div class="footer-wrap">
    <div class="footer-logo">🌿</div>
    <div class="footer-text">
        <strong>NeuralLeaf</strong> &nbsp;·&nbsp;
        Built with <strong>ResNet-18</strong> + <strong>GPT-4o</strong> &nbsp;·&nbsp;
        Dataset: <strong>PlantVillage</strong><br>
        Final Year Project &nbsp;·&nbsp; Made with ❤️ using Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
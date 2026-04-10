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
#  CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
 
/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
 
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
 
/* ── Hide chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
 
/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background: #f4fbf6 !important;
}
 
/* ── Layout ── */
.block-container {
    padding: 2rem 2.5rem 5rem !important;
    max-width: 1060px !important;
}
 
/* ── Tokens ── */
:root {
    --g700: #15803d;
    --g600: #16a34a;
    --g500: #22c55e;
    --g100: #dcfce7;
    --g50:  #f0fdf4;
    --text1: #111827;
    --text2: #374151;
    --text3: #6b7280;
    --border: rgba(34, 197, 94, 0.25);
    --card-bg: #ffffff;
    --shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
    --radius: 12px;
}
 
/* ═══════ HERO ═══════ */
.hero {
    text-align: center;
    padding: 2.8rem 1rem 2rem;
}
.hero-icon { font-size: 2.6rem; line-height: 1; margin-bottom: 0.7rem; display: block; }
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text1);
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
    line-height: 1.15;
}
.hero-title span { color: var(--g600); }
.hero-desc {
    font-size: 0.97rem;
    color: var(--text3);
    font-weight: 400;
    max-width: 460px;
    margin: 0 auto 1.5rem;
    line-height: 1.65;
}
.badge-row { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
.badge {
    background: var(--g50);
    border: 1px solid var(--border);
    color: var(--g700);
    font-size: 0.74rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #86efac, transparent);
    margin: 0.25rem 0 2rem;
    border: none;
}
 
/* ═══════ STEP HEADER ═══════ */
.step-hd { display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem; }
.step-num {
    width: 24px; height: 24px;
    background: var(--g600); color: #fff;
    border-radius: 50%;
    font-size: 0.7rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-label {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--g600);
}
 
/* ═══════ UPLOAD ═══════ */
.upload-card {
    background: var(--card-bg);
    border: 1.5px dashed #86efac;
    border-radius: var(--radius);
    padding: 1.8rem 1.5rem 1.2rem;
    text-align: center;
    box-shadow: var(--shadow);
    margin-bottom: 0.6rem;
}
.upload-card-icon { font-size: 2rem; display: block; margin-bottom: 0.5rem; }
.upload-card-title { font-size: 0.97rem; font-weight: 600; color: var(--text1); margin-bottom: 0.25rem; }
.upload-card-sub   { font-size: 0.82rem; color: var(--text3); margin-bottom: 0.7rem; }
.fmt-row { display: flex; justify-content: center; gap: 5px; flex-wrap: wrap; }
.fmt {
    background: var(--g50); border: 1px solid var(--border);
    color: var(--g700); font-size: 0.67rem; font-weight: 600;
    padding: 2px 8px; border-radius: 4px; letter-spacing: 0.4px;
}
 
/* Streamlit file uploader overrides */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploaderDropzone"] {
    background: var(--g50) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
 
/* ═══════ CARDS ═══════ */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem;
    box-shadow: var(--shadow);
}
.card-title {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 1.4px; text-transform: uppercase;
    color: var(--text3); margin-bottom: 0.85rem;
}
.img-meta { font-size: 0.75rem; color: var(--text3); text-align: center; margin-top: 0.6rem; }
 
/* ── Result pill ── */
.result-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--g600); color: #fff;
    font-size: 0.93rem; font-weight: 600;
    padding: 8px 18px; border-radius: 999px;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 10px rgba(22, 163, 74, 0.25);
}
.result-pill.warn {
    background: #d97706;
    box-shadow: 0 2px 10px rgba(217, 119, 6, 0.25);
}
 
/* ── Status line ── */
.status-line {
    font-size: 0.84rem; font-weight: 500;
    padding: 7px 12px; border-radius: 7px; margin-bottom: 1rem;
}
.status-line.ok  { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
.status-line.bad { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
 
/* ── Confidence bar ── */
.conf-wrap { margin-bottom: 1rem; }
.conf-hd { display: flex; justify-content: space-between; margin-bottom: 5px; }
.conf-lbl { font-size: 0.77rem; color: var(--text3); font-weight: 500; }
.conf-val { font-size: 0.84rem; font-weight: 700; color: var(--g600); }
.conf-track { height: 6px; background: #d1fae5; border-radius: 999px; overflow: hidden; }
.conf-fill  { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 999px; }
 
/* ── Metric pills ── */
.metric-row { display: flex; gap: 8px; margin-top: 0.9rem; }
.mpill {
    flex: 1; background: var(--g50);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 0.65rem 0.5rem; text-align: center;
}
.mpill-val { display: block; font-size: 1rem; font-weight: 700; color: var(--text1); }
.mpill-lbl { display: block; font-size: 0.63rem; color: var(--text3); font-weight: 500;
             text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }
 
/* ═══════ REMEDY ═══════ */
.remedy-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--g500);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1.4rem 1.5rem;
    font-size: 0.9rem; line-height: 1.8;
    color: var(--text2);
    box-shadow: var(--shadow);
}
 
/* ═══════ TABS ═══════ */
[data-testid="stTabs"] [role="tablist"] {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 7px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: var(--text3) !important;
    padding: 5px 14px !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: var(--g50) !important;
    color: var(--g700) !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stTabContent"] { padding-top: 1rem !important; }
 
/* ═══════ DOWNLOAD BUTTON ═══════ */
[data-testid="stDownloadButton"] > button {
    background: var(--g600) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 2px 10px rgba(22,163,74,0.2) !important;
    transition: opacity 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover { opacity: 0.88 !important; }
 
/* ═══════ ALERT BOXES ═══════ */
[data-testid="stAlert"] { border-radius: 9px !important; font-size: 0.87rem !important; }
 
/* ═══════ SIDEBAR ═══════ */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border) !important;
}
.sb-brand {
    text-align: center;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.sb-brand-icon { font-size: 1.9rem; display: block; margin-bottom: 0.3rem; }
.sb-brand-name { font-size: 1.05rem; font-weight: 700; color: var(--text1); }
.sb-brand-sub  { font-size: 0.73rem; color: var(--text3); margin-top: 1px; }
.sb-sec {
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: var(--g600); margin: 1.1rem 0 0.45rem;
}
.sb-row {
    display: flex; align-items: flex-start; gap: 8px;
    font-size: 0.81rem; color: var(--text2);
    padding: 4px 0;
    border-bottom: 1px solid #f3f4f6;
    line-height: 1.4;
}
.sb-row:last-child { border-bottom: none; }
.sb-row .ic { flex-shrink: 0; font-size: 0.88rem; margin-top: 1px; }
 
/* ═══════ MARKDOWN CONTRAST FIXES ═══════ */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--text2) !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}
 
/* ═══════ SPACING FIXES ═══════ */
div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }
.element-container:empty { display: none !important; }
[data-testid="stFileUploader"] > div:first-child { margin-bottom: 0 !important; }
[data-testid="column"] { padding: 0 0.35rem !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }
 
/* ═══════ FOOTER ═══════ */
.footer {
    text-align: center;
    border-top: 1px solid var(--border);
    padding: 1.8rem 0 0.25rem;
    margin-top: 2.5rem;
    font-size: 0.77rem;
    color: var(--text3);
    line-height: 1.9;
}
.footer strong { color: var(--g600); font-weight: 600; }
</style>
""", unsafe_allow_html=True)
 
 
# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-icon">🌿</span>
        <div class="sb-brand-name">NeuralLeaf</div>
        <div class="sb-brand-sub">Plant Disease AI</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown('<div class="sb-sec">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-row"><span class="ic">🎓</span> Final Year Project</div>
    <div class="sb-row"><span class="ic">🤖</span> Deep Learning + GPT-4o</div>
    <div class="sb-row"><span class="ic">🌾</span> Agriculture AI Domain</div>
    """, unsafe_allow_html=True)
 
    st.markdown('<div class="sb-sec">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-row"><span class="ic">🧠</span> Architecture: ResNet-18</div>
    <div class="sb-row"><span class="ic">🏷️</span> Classes: 15 Disease Labels</div>
    <div class="sb-row"><span class="ic">🖼️</span> Input: 224 × 224 px</div>
    <div class="sb-row"><span class="ic">📦</span> Dataset: PlantVillage</div>
    """, unsafe_allow_html=True)
 
    st.markdown('<div class="sb-sec">Supported Plants</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-row"><span class="ic">🫑</span> Pepper Bell</div>
    <div class="sb-row"><span class="ic">🥔</span> Potato</div>
    <div class="sb-row"><span class="ic">🍅</span> Tomato</div>
    """, unsafe_allow_html=True)
 
    st.markdown('<div class="sb-sec">How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-row"><span class="ic">1️⃣</span> Upload a clear leaf photo</div>
    <div class="sb-row"><span class="ic">2️⃣</span> Wait for AI prediction</div>
    <div class="sb-row"><span class="ic">3️⃣</span> Read the remedy advice</div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 Use a clear, well-lit photo of a single leaf for best accuracy.")
 
 
# ============================================================
#  HERO
# ============================================================
st.markdown("""
<div class="hero">
    <span class="hero-icon">🌿</span>
    <div class="hero-title">Neural<span>Leaf</span></div>
    <p class="hero-desc">
        Upload a plant leaf photo for an instant AI-powered disease
        diagnosis with expert treatment advice.
    </p>
    <div class="badge-row">
        <span class="badge">🧠 ResNet-18</span>
        <span class="badge">⚡ GPT-4o Remedy</span>
        <span class="badge">🌾 PlantVillage</span>
        <span class="badge">🏷️ 15 Classes</span>
    </div>
</div>
<div class="divider"></div>
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
<div class="step-hd">
    <div class="step-num">1</div>
    <div class="step-label">Upload Leaf Image</div>
</div>
<div class="upload-card">
    <span class="upload-card-icon">🍃</span>
    <div class="upload-card-title">Drop your leaf photo here</div>
    <div class="upload-card-sub">Drag &amp; drop, or click Browse to upload</div>
    <div class="fmt-row">
        <span class="fmt">JPG</span>
        <span class="fmt">PNG</span>
        <span class="fmt">JPEG</span>
        <span class="fmt">WEBP</span>
    </div>
</div>
""", unsafe_allow_html=True)
 
uploaded_file = st.file_uploader(
    label="Upload leaf image",
    type=["jpg", "png", "jpeg", "webp"],
    help="Supported: JPG, PNG, JPEG, WEBP",
    label_visibility="collapsed"
)
 
 
# ============================================================
#  STEPS 2 & 3 — RESULTS
# ============================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
 
    # Inference (unchanged)
    with torch.no_grad():
        outputs    = model(img_tensor)
        prob       = torch.softmax(outputs, dim=1)
        predicted  = torch.argmax(prob, dim=1)
        confidence = prob[0][predicted.item()].item() * 100
 
    label     = classes[predicted.item()]
    healthy   = is_healthy(label)
    fill_pct  = int(confidence)
    pill_cls  = "result-pill" if healthy else "result-pill warn"
    strip_cls = "status-line ok" if healthy else "status-line bad"
    status_ico = "✅" if healthy else "⚠️"
    status_msg = "Plant appears healthy — no disease detected." if healthy \
                 else "Disease detected — expert remedy provided below."
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    st.markdown("""
    <div class="step-hd">
        <div class="step-num">2</div>
        <div class="step-label">Diagnosis Results</div>
    </div>
    """, unsafe_allow_html=True)
 
    col_img, col_res = st.columns([1, 1.15], gap="medium")
 
    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Uploaded Leaf</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f'<p class="img-meta">{uploaded_file.name} &nbsp;·&nbsp; '
            f'{image.width} × {image.height} px</p>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
 
    with col_res:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">AI Prediction</div>', unsafe_allow_html=True)
 
        st.markdown(f'<div class="{pill_cls}">{status_ico} {label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{strip_cls}">{status_msg}</div>', unsafe_allow_html=True)
 
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-hd">
                <span class="conf-lbl">Model Confidence</span>
                <span class="conf-val">{confidence:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{fill_pct}%"></div>
            </div>
        </div>
        <div class="metric-row">
            <div class="mpill">
                <span class="mpill-val">{confidence:.1f}%</span>
                <span class="mpill-lbl">Confidence</span>
            </div>
            <div class="mpill">
                <span class="mpill-val">{"✅" if healthy else "⚠️"}</span>
                <span class="mpill-lbl">Status</span>
            </div>
            <div class="mpill">
                <span class="mpill-val">15</span>
                <span class="mpill-lbl">Classes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown('</div>', unsafe_allow_html=True)
 
    # ── Step 3 ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-hd">
        <div class="step-num">3</div>
        <div class="step-label">AI Remedy &amp; Recommendations</div>
    </div>
    """, unsafe_allow_html=True)
 
    with st.spinner("Generating expert remedy…"):
        remedy = generate_remedy(label)
 
    tab_remedy, tab_tips, tab_raw = st.tabs(
        ["📋 Full Remedy", "📌 Quick Tips", "🗒️ Raw Output"]
    )
 
    with tab_remedy:
        st.markdown(
            f'<div class="remedy-card">{remedy.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )
 
    with tab_tips:
        if healthy:
            st.success("Your plant looks healthy! Keep it that way:")
            st.markdown("""
- ✅ Water consistently, avoid waterlogging
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
- 🟡 Apply appropriate fungicide or pesticide
- 🟡 Reduce overhead watering; water at the base
- 🟢 Monitor surrounding plants for spread
- 🟢 Consult a local agricultural officer if severe
            """)
 
    with tab_raw:
        st.code(remedy, language="markdown")
 
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
<div class="footer">
    🌿 <strong>NeuralLeaf</strong> &nbsp;·&nbsp;
    Built with <strong>ResNet-18</strong> + <strong>GPT-4o</strong> &nbsp;·&nbsp;
    Dataset: <strong>PlantVillage</strong><br>
    Final Year Project &nbsp;·&nbsp; Made with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
 
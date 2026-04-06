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
#  GLOBAL CSS
# ============================================================
st.markdown("""
<style>
/* ---------- Streamlit chrome overrides ---------- */
#MainMenu, footer, header {visibility: hidden;}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ---------- Design tokens ---------- */
:root {
    --green-primary:  #00C897;
    --green-dark:     #00956E;
    --green-glow:     rgba(0, 200, 151, 0.18);
    --bg-card:        rgba(255, 255, 255, 0.04);
    --bg-card-hover:  rgba(255, 255, 255, 0.07);
    --border-subtle:  rgba(255, 255, 255, 0.09);
    --text-muted:     #9CA3AF;
    --radius-lg:      18px;
    --radius-md:      12px;
    --shadow-card:    0 4px 32px rgba(0,0,0,0.35);
}

/* ---------- Hero ---------- */
.hero-wrap {
    text-align: center;
    padding: 2.4rem 1rem 1.6rem;
}
.hero-logo {
    font-size: 3.6rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(100deg, #00C897 20%, #38bdf8 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    margin-bottom: 1.2rem;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.badge {
    background: var(--green-glow);
    border: 1px solid var(--green-primary);
    color: var(--green-primary);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}

/* ---------- Divider ---------- */
.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--green-primary), transparent);
    border: none;
    margin: 0.5rem 0 2rem;
    opacity: 0.5;
}

/* ---------- Section label ---------- */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--green-primary);
    margin-bottom: 0.6rem;
}

/* ---------- Glass card ---------- */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.6rem 1.4rem;
    box-shadow: var(--shadow-card);
    margin-bottom: 1rem;
}

/* ---------- Upload hint ---------- */
.upload-hint {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

/* ---------- Disease result chip ---------- */
.result-chip {
    display: inline-block;
    background: linear-gradient(135deg, #00C897, #0575E6);
    color: #fff;
    font-size: 1.15rem;
    font-weight: 700;
    padding: 10px 22px;
    border-radius: 999px;
    margin-bottom: 1.1rem;
    letter-spacing: 0.2px;
}

/* ---------- Confidence bar container ---------- */
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.conf-label { font-size: 0.82rem; color: var(--text-muted); }
.conf-value { font-size: 0.95rem; font-weight: 700; color: var(--green-primary); }

/* ---------- Remedy block ---------- */
.remedy-body {
    background: rgba(0, 200, 151, 0.06);
    border-left: 3px solid var(--green-primary);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #D1FAE5;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: rgba(15, 25, 20, 0.97) !important;
    border-right: 1px solid var(--border-subtle);
}
.sb-logo { font-size: 2rem; text-align: center; margin-bottom: 0.2rem; }
.sb-title {
    text-align: center;
    font-weight: 800;
    font-size: 1.2rem;
    background: linear-gradient(90deg, #00C897, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.sb-section {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--green-primary);
    margin: 1.2rem 0 0.4rem;
}
.sb-item {
    font-size: 0.88rem;
    color: #CBD5E1;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-subtle);
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
    padding: 1.8rem 0 0.5rem;
    border-top: 1px solid var(--border-subtle);
    margin-top: 2.5rem;
}
.footer span { color: var(--green-primary); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-logo">🌿</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">NeuralLeaf AI</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item">🎓 Final Year Project</div>
    <div class="sb-item">🤖 Deep Learning + GPT-4o</div>
    <div class="sb-item">🌾 Agriculture AI Domain</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item">🧠 Architecture: ResNet-18</div>
    <div class="sb-item">🏷️ Classes: 15 Disease Labels</div>
    <div class="sb-item">🖼️ Input Size: 224 × 224 px</div>
    <div class="sb-item">📦 Dataset: PlantVillage</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Supported Plants</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item">🫑 Pepper Bell</div>
    <div class="sb-item">🥔 Potato</div>
    <div class="sb-item">🍅 Tomato</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-item">1️⃣ Upload a leaf image</div>
    <div class="sb-item">2️⃣ Wait for prediction</div>
    <div class="sb-item">3️⃣ Read AI remedy advice</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📌 Best results with clear, well-lit leaf photos against a plain background.", icon="💡")


# ============================================================
#  HERO SECTION
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-logo">🌿</div>
    <p class="hero-title">NeuralLeaf</p>
    <p class="hero-sub">Upload a plant leaf image — get an instant AI-powered disease diagnosis and actionable remedy.</p>
    <div class="badge-row">
        <span class="badge">🧠 ResNet-18</span>
        <span class="badge">⚡ GPT-4o Remedy</span>
        <span class="badge">🌾 PlantVillage Dataset</span>
        <span class="badge">15 Disease Classes</span>
    </div>
</div>
<hr class="fancy-divider"/>
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

# Helper: is the detected class a healthy label?
def is_healthy(label: str) -> bool:
    return "healthy" in label.lower()


# ============================================================
#  UPLOAD SECTION
# ============================================================
st.markdown('<p class="section-label">📤 Step 1 — Upload Leaf Image</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "png", "jpeg", "webp"],
    help="Supported: JPG, PNG, JPEG, WEBP"
)
st.markdown(
    '<p class="upload-hint">📌 For best accuracy use a clear, well-lit photo of a single leaf.</p>',
    unsafe_allow_html=True
)


# ============================================================
#  RESULTS SECTION
# ============================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    # ---- Inference (unchanged logic) ----
    with torch.no_grad():
        outputs   = model(img_tensor)
        prob      = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(prob, dim=1)
        confidence = prob[0][predicted.item()].item() * 100

    label   = classes[predicted.item()]
    healthy = is_healthy(label)
    status_icon  = "✅" if healthy else "⚠️"
    status_color = "#00C897" if healthy else "#F97316"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">🔬 Step 2 — Diagnosis Results</p>', unsafe_allow_html=True)

    col_img, col_res = st.columns([1, 1.15], gap="large")

    # ---- Left column: image ----
    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Uploaded Leaf</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f'<p style="text-align:center;color:#9CA3AF;font-size:0.8rem;margin-top:6px;">'
            f'{uploaded_file.name} &nbsp;|&nbsp; {image.width}×{image.height} px</p>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Right column: result ----
    with col_res:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">AI Prediction</p>', unsafe_allow_html=True)

        # Disease chip
        st.markdown(
            f'<div class="result-chip">{status_icon} {label}</div>',
            unsafe_allow_html=True
        )

        # Status badge
        status_text = "Plant appears healthy 🎉" if healthy else "Disease detected — see remedy below"
        st.markdown(
            f'<p style="color:{status_color};font-weight:600;font-size:0.9rem;margin-bottom:1rem;">'
            f'{status_text}</p>',
            unsafe_allow_html=True
        )

        # Confidence metric + progress bar
        st.markdown(
            '<div class="conf-row">'
            '<span class="conf-label">Model Confidence</span>'
            f'<span class="conf-value">{confidence:.1f}%</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.progress(int(confidence))

        # Extra metrics row
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{confidence:.1f}%")
        m2.metric("Status", "Healthy ✅" if healthy else "Diseased ⚠️")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Remedy section ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">💊 Step 3 — AI Remedy & Recommendations</p>', unsafe_allow_html=True)

    with st.spinner("🤖 Generating expert remedy with GPT-4o…"):
        remedy = generate_remedy(label)

    tab_remedy, tab_tips, tab_raw = st.tabs(["📋 Full Remedy", "📌 Quick Tips", "🗒️ Raw Output"])

    with tab_remedy:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="remedy-body">{remedy.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_tips:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if healthy:
            st.success("Your plant looks healthy! Here are general tips to keep it that way:")
            st.markdown("""
            - ✅ Water consistently but avoid waterlogging  
            - ✅ Ensure good sunlight and air circulation  
            - ✅ Inspect leaves weekly for early signs of disease  
            - ✅ Use balanced fertiliser during the growing season  
            - ✅ Rotate crops seasonally to prevent soil-borne diseases  
            """)
        else:
            st.warning(f"Quick action checklist for **{label}**:")
            st.markdown("""
            - 🔴 Isolate the affected plant immediately  
            - 🔴 Remove and destroy visibly infected leaves  
            - 🟡 Apply appropriate fungicide / pesticide  
            - 🟡 Reduce overhead watering; water at the base  
            - 🟢 Monitor surrounding plants for spread  
            - 🟢 Consult a local agricultural extension officer if severe  
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_raw:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.code(remedy, language="markdown")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Download remedy ----
    st.download_button(
        label="⬇️ Download Remedy Report",
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
    🌿 <span>NeuralLeaf</span> &nbsp;|&nbsp;
    Built with <span>ResNet-18</span> + <span>GPT-4o</span> &nbsp;|&nbsp;
    Dataset: <span>PlantVillage</span> &nbsp;|&nbsp;
    Final Year Project &nbsp;|&nbsp;
    Made with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
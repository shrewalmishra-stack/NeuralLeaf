import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from openai import OpenAI
import os

# ------------------ API SETUP ------------------
import streamlit as st
from openai import OpenAI
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

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="NeuralLeaf", page_icon="🌿", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
/* Background Gradient */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Title */
.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    background: linear-gradient(90deg, #00F260, #0575E6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #CCCCCC;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(0,255,200,0.2);
    margin-bottom: 20px;
}

/* Highlight box */
.highlight {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    padding: 12px;
    border-radius: 10px;
    color: black;
    font-weight: bold;
}

/* Remedy box */
.remedy {
    background: rgba(0, 255, 150, 0.08);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">🌿 NeuralLeaf</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart AI for Plant Health Diagnosis</div>', unsafe_allow_html=True)

st.write("https://drive.google.com/uc?id=1llTeglJuXxsudW_1N11_SMXtgfTIZLXA")

# imports ...

import urllib.request

MODEL_URL = ""
MODEL_PATH = "plant_model.pth"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ------------------ LOAD MODEL ------------------
model = models.resnet18()
# ------------------ LOAD MODEL ------------------
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 15)
model.load_state_dict(torch.load("plant_model.pth", map_location=torch.device('cpu')))
model.eval()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------ CLASS NAMES ------------------
classes = [
    "Pepper Bell Healthy", "Potato Early Blight", "Potato Late Blight",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Mosaic Virus", "Tomato Yellow Leaf Curl Virus",
    "Tomato Healthy", "Pepper Bell Bacterial Spot", "Potato Healthy"
]

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📸 Upload a leaf image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", width=350)
        st.markdown('</div>', unsafe_allow_html=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        prob = torch.softmax(outputs, dim=1)
        confidence = prob[0][predicted.item()].item() * 100

    label = classes[predicted.item()]

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f'<div class="highlight">🌿 {label}</div>', unsafe_allow_html=True)
        st.write(f"📊 Confidence: {confidence:.2f}%")


        # AI Remedy
        with st.spinner("🤖 Generating AI-based remedy..."):
            remedy = generate_remedy(label)

    st.markdown("### 💊 Recommended Solution")
    st.markdown('<div class="remedy">',unsafe_allow_html=True)
    st.write(remedy)
    st.markdown('</div>',unsafe_allow_html=True)
      

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<center>⚡ Powered by NeuralLeaf AI | Built with Deep Learning + AI</center>", unsafe_allow_html=True)
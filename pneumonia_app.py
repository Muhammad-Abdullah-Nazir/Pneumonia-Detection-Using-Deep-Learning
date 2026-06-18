import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

# ---- MODEL ARCHITECTURES ----

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


def load_resnet():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    return model


# ---- LOAD MODELS ----

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = PneumoniaCNN()
    cnn.load_state_dict(torch.load("best_cnn_scratch.pth", map_location=device))
    cnn.to(device).eval()

    resnet = load_resnet()
    resnet.load_state_dict(torch.load("best_resnet50.pth", map_location=device))
    resnet.to(device).eval()

    return cnn, resnet, device


# ---- PREPROCESS IMAGE ----

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# ---- PREDICT ----

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = output.argmax(dim=1).item()
    return pred, probs[0].item(), probs[1].item()


# ---- UI HEADER ----

st.markdown("""
    <h1 style="text-align:center; color:#1F4E79;">🫁 Pneumonia Detection System</h1>
    <p style="text-align:center; color:#555; font-size:16px;">
        Upload a chest X-ray image to detect Pneumonia using Deep Learning
    </p>
    <hr style="border: 1px solid #1F4E79;">
""", unsafe_allow_html=True)


# ---- SIDEBAR ----

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["ResNet-50 (Recommended)", "CNN from Scratch"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("""
| Metric | CNN | ResNet |
|--------|-----|--------|
| Accuracy | 83.49% | 83.65% |
| ROC-AUC | 0.914 | 0.963 |
| Pneumonia Recall | 96% | 99% |
| Missed Sick | 16 | **2** |
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
**Project:** ANN & Deep Learning
**Semester:** BS AI — 6th
**Team:**
- Inam-Ur-Rehman (076)
- M. Abdullah Nazir (078)

**Supervisor:** Dr Muhammad Gufran Khan
""")


# ---- FILE UPLOAD ----

st.markdown("### 📤 Upload Chest X-Ray Image")
uploaded_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🖼️ Uploaded X-Ray")
        st.image(image, width=400)

    with col2:
        st.markdown("#### 🔬 Analysis Result")

        with st.spinner("Analyzing X-ray..."):
            cnn_model, resnet_model, device = load_models()
            img_tensor = preprocess(image)

            if "ResNet" in model_choice:
                pred, prob_normal, prob_pneumonia = predict(resnet_model, img_tensor, device)
                model_name = "ResNet-50"
            else:
                pred, prob_normal, prob_pneumonia = predict(cnn_model, img_tensor, device)
                model_name = "CNN from Scratch"

        if pred == 1:
            st.markdown("""
            <div style="background:#ffe6e6; padding:20px; border-radius:10px;
                        border-left:6px solid #e74c3c;">
                <h3 style="color:#c0392b;">🚨 Pneumonia Detected</h3>
                <p>The model identified signs of pneumonia in this X-ray.<br><br>
                <b>Please consult a qualified doctor for medical advice.</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#e6ffe6; padding:20px; border-radius:10px;
                        border-left:6px solid #27ae60;">
                <h3 style="color:#1e6b3c;">✅ Lungs Appear Normal</h3>
                <p>No signs of pneumonia detected in this X-ray.<br><br>
                <b>Always confirm with a medical professional.</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**Model used:** {model_name}")
        st.markdown("---")
        st.markdown("**🟢 Normal**")
        st.progress(prob_normal)
        st.markdown(f"**{prob_normal*100:.1f}%** confidence")

        st.markdown("**🔴 Pneumonia**")
        st.progress(prob_pneumonia)
        st.markdown(f"**{prob_pneumonia*100:.1f}%** confidence")

    st.markdown("---")
    st.warning("⚠️ This tool is for academic purposes only. Not a substitute for professional medical diagnosis.")

else:
    st.info("👆 Upload a chest X-ray image above to get started.")
    st.markdown("""
    ### 💡 How to use:
    1. Select a model from the sidebar
    2. Upload a chest X-ray (JPG or PNG)
    3. Wait a few seconds
    4. See the prediction and confidence score

    ### 🧪 Test images are here:
    - `chest_xray/test/NORMAL/` — healthy lungs
    - `chest_xray/test/PNEUMONIA/` — infected lungs
    """)
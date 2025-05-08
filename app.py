# This is your full diabetic retinopathy detection app with auto-download model
# Clean UI, Grad-CAM, SQLite logging, and downloadable A4 report included.

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
from datetime import datetime
import sqlite3
import cv2
from fpdf import FPDF
import os
import gdown

# Auto-download model from Google Drive if not present
model_path = "dr_vgg16_final_finetuned3.pth"
drive_file_id = "1zvdFnMcCw-8p4jqWRQe5G0lpz8qDEooc"
model_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

class_names = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']
severity_description = {
    "Healthy": "No signs of diabetic retinopathy detected.",
    "Mild DR": "Microaneurysms present. Regular monitoring recommended.",
    "Moderate DR": "Increased microaneurysms and hemorrhages. Medical treatment advised.",
    "Proliferate DR": "New abnormal blood vessels forming. Requires urgent medical attention.",
    "Severe DR": "Significant retinal damage. High risk of vision loss without treatment."
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 5)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        x.requires_grad_()
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()
        gradients = self.gradients.detach()[0]
        activations = self.activations.detach()[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()

target_layer = model.features[26]
cam_generator = GradCAM(model, target_layer)

def apply_colormap_on_image(org_img, activation_map, colormap_name=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), colormap_name)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, org_img.size)
    overlay = np.array(org_img.convert("RGB")) * 0.6 + heatmap * 0.4
    return Image.fromarray(np.uint8(overlay))

def generate_pdf_report(name, age, date, prediction, probs, class_names, original_img, overlay_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Diabetic Retinopathy Diagnostic Report", ln=True, align="C")

    pdf.set_draw_color(0, 0, 0)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "", ln=True)
    pdf.cell(190, 8, "Patient Information", ln=True, border="B")
    pdf.set_font("Arial", "", 11)
    pdf.cell(95, 8, f"Name: {name}", border=0)
    pdf.cell(95, 8, f"Age: {age}", ln=True, border=0)
    pdf.cell(95, 8, f"Date: {date}", ln=True, border=0)

    pdf.cell(0, 10, "", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "Prediction Summary", ln=True, border="B")
    pdf.set_font("Arial", "", 11)
    pdf.cell(190, 8, f"Predicted Class: {prediction}", ln=True)
    pdf.multi_cell(0, 8, f"Interpretation: {severity_description[prediction]}", border=0)

    pdf.cell(0, 10, "", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "Class Probabilities", ln=True, border="B")
    pdf.set_font("Arial", "", 11)
    for cls, p in zip(class_names, probs):
        pdf.cell(95, 8, f"{cls}: {p*100:.2f}%", ln=True)

    pdf.cell(0, 10, "", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "Retinal Image Analysis", ln=True, border="B")
    original_img.save("orig_temp.jpg")
    overlay_img.save("overlay_temp.jpg")
    pdf.ln(5)
    pdf.set_font("Arial", "I", 11)
    pdf.cell(75, 8, "Original Image", border=0, ln=0, align="C")
    pdf.cell(40, 8, "", border=0, ln=0)
    pdf.cell(75, 8, "Grad-CAM Overlay", border=0, ln=1, align="C")
    pdf.image("orig_temp.jpg", x=25, y=pdf.get_y(), w=75)
    pdf.image("overlay_temp.jpg", x=110, y=pdf.get_y(), w=75)
    return io.BytesIO(pdf.output(dest="S").encode("latin1"))

conn = sqlite3.connect("retinopathy_predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        prediction TEXT,
        probs TEXT,
        date TEXT,
        image BLOB,
        heatmap BLOB
    )
''')
conn.commit()

st.sidebar.header("🔍 Prediction History")
history = c.execute("SELECT id, name, age, prediction, probs, date FROM predictions ORDER BY id DESC").fetchall()
for rid, name, age, pred, probs, date in history[:5]:
    st.sidebar.markdown(f"{name}, {age}y – **{pred}** on {date}")

with st.form("predict_form", clear_on_submit=False):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    upload = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Run Prediction")

if submit:
    if not name or not upload:
        st.warning("Please enter name and upload an image.")
    else:
        image = Image.open(upload).convert("RGB")
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits[0], dim=0).cpu().numpy()
        top_idx = int(probs.argmax())
        prediction = class_names[top_idx]
        cam = cam_generator.generate(x, top_idx)
        overlay = apply_colormap_on_image(image, cam)

        st.subheader(f"Prediction: *{prediction}* ({probs[top_idx]*100:.1f}%)")
        st.markdown("*Top 3 Class Probabilities:*")
        top3 = probs.argsort()[-3:][::-1]
        for idx in top3:
            st.write(f"- {class_names[idx]}: {probs[idx]*100:.2f}%")

        st.markdown("### Output Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width=350)
        with col2:
            st.image(overlay, caption="Grad-CAM Overlay", width=350)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_buf = io.BytesIO(); image.save(img_buf, format="PNG")
        hm_buf = io.BytesIO(); overlay.save(hm_buf, format="PNG")
        c.execute("INSERT INTO predictions (name, age, prediction, probs, date, image, heatmap) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (name, age, prediction, ",".join(f"{v:.4f}" for v in probs), now, img_buf.getvalue(), hm_buf.getvalue()))
        conn.commit()

        pdf_buf = generate_pdf_report(name, age, now, prediction, probs, class_names, image, overlay)
        st.download_button("📄 Download Report (PDF)", data=pdf_buf.getvalue(), file_name=f"{name}_DR_Report.pdf", mime="application/pdf")


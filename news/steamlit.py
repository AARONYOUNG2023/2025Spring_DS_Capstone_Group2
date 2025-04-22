import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from combined_model import ChestXRayReportGenerator

# ----------------------------------------------------------- #
# Page & dark‑theme styling
# ----------------------------------------------------------- #
st.set_page_config(page_title="Chest X‑ray Report Generator", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Open Sans', sans-serif; background:#121212; color:#fff; }
    .stButton>button, .stDownloadButton>button { border:1px solid #fff !important; background:#1e1e1e !important; color:#fff; }
    .stButton>button:hover, .stDownloadButton>button:hover { background:#fff !important; color:#000 !important; }
    .report-nav { background:#1f1f1f; padding:10px 0; margin:-2rem -2rem 2rem -2rem; text-align:center; border-bottom:1px solid #444; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="report-nav"><h1>🩺 Chest X‑ray Report Generator</h1></div>', unsafe_allow_html=True)

# ----------------------------------------------------------- #
# Load model only once
# ----------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading V&L model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = ChestXRayReportGenerator().to(device).eval()
    return mdl, device

model, device = load_model()

# ----------------------------------------------------------- #
# Helper: preprocess PIL → tensor
# ----------------------------------------------------------- #
def preprocess(image: Image.Image) -> torch.Tensor:
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return tfms(image).unsqueeze(0).to(device)

# ----------------------------------------------------------- #
# UI
# ----------------------------------------------------------- #
st.write("Upload a chest X‑ray image and generate an AI‑assisted radiology report.")

file = st.file_uploader("Choose PNG / JPG", type=["png", "jpg", "jpeg"])

if file:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded X‑ray", use_column_width=True)

        prompt = st.text_input("Prompt prefix", value="The chest X‑ray shows")
        num_masks = st.slider("Mask tokens (length of completion)",
                              min_value=5, max_value=30, value=15)

        if st.button("🔍 Generate report"):
            with st.spinner("Generating…"):
                img_tensor = preprocess(img)
                report = model.generate_text(
                    images=img_tensor,
                    prompt_text=prompt,
                    num_mask_tokens=num_masks
                )

            st.success("Report generated!")
            st.markdown("### Generated Report")
            st.write(report)

            st.download_button(
                "Download as .txt",
                data=report,
                file_name="xray_report.txt",
                mime="text/plain"
            )

    except Exception as err:
        st.error(f"Failed to process the image: {err}")
else:
    st.info("👈 Upload an X‑ray image to begin.")

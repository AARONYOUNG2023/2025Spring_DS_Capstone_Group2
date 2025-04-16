import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from combined_model import ChestXRayReportGenerator

#
st.set_page_config(page_title="Chest X-ray Report Generator", layout="centered")
custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans&display=swap');

        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            background-color: #121212;
            color: white;
            animation: fadeIn 1.2s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .stTextInput > div > div > input,
        .stTextArea > div > textarea,
        .stFileUploader > div,
        .stSlider > div > div > div {
            border: 1px solid white !important;
            background-color: #1e1e1e !important;
            color: white !important;
        }

        .stButton > button, .stDownloadButton > button {
            border: 1px solid white !important;
            background-color: #1e1e1e !important;
            color: white !important;
            transition: all 0.3s ease;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: white !important;
            color: black !important;
        }

        .report-nav {
            background-color: #1f1f1f;
            padding: 10px 20px;
            border-bottom: 1px solid #444;
            margin: -2rem -2rem 2rem -2rem;
            text-align: center;
        }
        .report-nav h1 {
            margin: 0;
            font-size: 1.6rem;
            color: #ffffff;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

#
st.markdown("""
<div class="report-nav">
    <h1>🩺 Chest X-ray Report Generator</h1>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXRayReportGenerator().to(device)
    model.eval()
    return model, device

model, device = load_model()

#
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

#
st.markdown("Upload a chest X-ray image and generate an AI-assisted radiology report using a vision-language model.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        prompt = st.text_input("Prompt (optional)", value="The chest X-ray reveals")
        num_mask_tokens = st.slider("Number of mask tokens", 5, 30, value=15)

        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                image_tensor = preprocess_image(image)
                result = model.generate_text(
                    images=image_tensor,
                    prompt_text=prompt,
                    num_mask_tokens=num_mask_tokens
                )

            st.success("Report generated successfully!")
            st.markdown("###Generated Report:")
            st.write(result)

            st.download_button(
                label="Download Report as TXT",
                data=result,
                file_name="xray_report.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Failed to process the image: {e}")
else:
    st.info("Please upload a chest X-ray image to get started.")



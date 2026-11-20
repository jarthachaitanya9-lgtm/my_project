import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Encoder-Decoder Transformer POC", layout="centered")
st.title("Plant Leaf Disease Diagnostic System")
st.caption("Architecture: Vision Transformer (Encoder) + Cross-Attention + Text Transformer (Decoder)")

@st.cache_resource
def load_model():
    model_path = "./saved_leaf_model"
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to("cpu")
    model.eval()
    return processor, model

processor, model = load_model()

remedy_guide = {
    "angular leaf spot": ("Bacterial / Fungal Lesions", "Apply copper-based bactericide/fungicide and remove infected lower leaves."),
    "bean rust": ("Fungal Infection (Uromyces appendiculatus)", "Use sulfur or chlorothalonil fungicide spray and improve air circulation."),
    "healthy": ("None (Healthy Foliage)", "No treatment required. Maintain regular watering and nutrient schedule."),
}

uploaded_file = st.file_uploader("Upload a plant leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image (Encoded as 16x16 Patches by ViT Encoder)", use_container_width=True)

    if st.button("Generate Diagnostic Report"):
        with st.spinner("ViT Encoder extracting visual patches & Text Decoder generating diagnosis..."):
            inputs = processor(images=image, return_tensors="pt").to("cpu")
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=40)
                output_text = processor.decode(generated_ids[0], skip_special_tokens=True)

        st.success("Autoregressive Text Decoder Output:")
        st.write(f"**{output_text.capitalize()}**")

        for key, (cause, action) in remedy_guide.items():
            if key in output_text.lower():
                st.info(f"**Pathogen / Cause:** {cause}\n\n**Recommended Action:** {action}")
                break

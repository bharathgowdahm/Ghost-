import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Ghost Eraser Pro", page_icon="👻", layout="wide")
st.title("👻 GHOST ERASER - Pro")
st.caption("Upload photo → Auto removes ghosts / shadows / light leaks. No brush library needed.")

uploaded = st.file_uploader("Upload your photo", type=["jpg","jpeg","png","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)
    h, w = img.shape[:2]

    # Resize for speed
    max_side = 800
    if max(h,w) > max_side:
        scale = max_side / max(h,w)
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
    else:
        img_small = img.copy()

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_container_width=True)
        mode = st.radio("Erase Mode", ["Auto Ghost Detect (1-click)", "Manual Mask Upload"])

        mask_file = None
        if mode == "Manual Mask Upload":
            st.info("Create a black image with WHITE paint over ghost area in your phone gallery, then upload it as mask")
            mask_file = st.file_uploader("Upload White-on-Black Mask (same size as photo)", type=["png","jpg"], key="mask")

    with col2:
        if st.button("✨ ERASE GHOST NOW", type="primary", use_container_width=True):
            with st.spinner("Erasing ghost..."):
                if mode == "Auto Ghost Detect (1-click)":
                    # Auto detect light ghosting / shadows
                    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
                    # Detect semi-transparent bright areas = ghosts
                    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                    # Also detect shadow edges
                    edges = cv2.Canny(gray, 30, 100)
                    mask = cv2.bitwise_or(mask, edges)
                    mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=2)
                else:
                    if mask_file is None:
                        st.warning("Upload a mask first")
                        st.stop()
                    mask_img = Image.open(mask_file).convert("L")
                    mask = np.array(mask_img)
                    mask = cv2.resize(mask, (img_small.shape[1], img_small.shape[0]))
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Inpaint
                img_cv = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
                inpainted = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_NS)
                result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                result = cv2.detailEnhance(result, sigma_s=10, sigma_r=0.15)

                # Upscale back to original size
                result_full = cv2.resize(result, (w, h))

                st.image(result_full, caption="Clean Result", use_container_width=True)
                result_pil = Image.fromarray(result_full)
                # Save for download
                from io import BytesIO
                buf = BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button("⬇️ Download HD", buf.getvalue(), "ghost_erased.png", "image/png", use_container_width=True)
                st.success("Ghost erased successfully!")
else:
    st.info("👆 Upload a photo to start. This version has ZERO heavy libraries, will not show 'Error installing requirements'")

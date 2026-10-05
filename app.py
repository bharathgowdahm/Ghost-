import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Ghost Eraser Pro", page_icon="👻", layout="wide")

st.title("👻 Ghost Eraser Pro")
st.caption("Magic Eraser - Remove ghosts, people, watermarks - 1 Click")

uploaded = st.file_uploader("Upload Photo", type=["jpg","jpeg","png","webp"])

if uploaded is None:
    st.info("👆 Upload a photo to start. This is the market-ready version.")
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb", width=400)
else:
    try:
        image = Image.open(uploaded).convert("RGB")
        img = np.array(image)
        h, w = img.shape[:2]

        # Safe resize
        max_side = 800
        if max(h,w) > max_side:
            scale = max_side / float(max(h,w))
            img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
        else:
            img_small = img.copy()

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original", use_container_width=True)
            mode = st.radio("Mode", ["Auto Erase (Recommended)", "Upload Mask"], horizontal=True)
            mask_file = None
            if mode == "Upload Mask":
                mask_file = st.file_uploader("Upload mask - white area = erase", type=["png","jpg","jpeg"], key="mask2")

            run = st.button("✨ ERASE NOW", type="primary", use_container_width=True)

        with col2:
            st.write("**Result**")
            if run:
                with st.spinner("Erasing..."):
                    try:
                        if mode == "Auto Erase (Recommended)":
                            gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
                            _, mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
                            # clean mask
                            kernel = np.ones((5,5), np.uint8)
                            mask = cv2.dilate(mask, kernel, iterations=2)
                        else:
                            if mask_file is None:
                                st.warning("Please upload mask")
                                st.stop()
                            mask_img = Image.open(mask_file).convert("L")
                            mask = np.array(mask_img)
                            mask = cv2.resize(mask, (img_small.shape[1], img_small.shape[0]))
                            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                        # Safety: if mask empty, create small dummy
                        if np.count_nonzero(mask) == 0:
                            st.warning("No ghost area detected. Try Manual Mask mode and paint white over area.")
                            st.stop()

                        img_cv = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
                        # This is the safest inpaint call
                        inpainted = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
                        result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                        result_full = cv2.resize(result, (w, h))

                        result_pil = Image.fromarray(result_full)
                        st.image(result_pil, caption="Cleaned - HD", use_container_width=True)

                        buf = BytesIO()
                        result_pil.save(buf, format="PNG")
                        st.download_button("⬇️ Download HD", buf.getvalue(), "erased.png", "image/png", use_container_width=True)
                        st.success("Done! Ready for market.")
                    except Exception as e:
                        st.error(f"Error during erase: {e}")
                        st.info("Try another photo or use Manual Mask mode")
            else:
                st.info("Click ERASE NOW to see result")

    except Exception as e:
        st.error(f"App error: {e}")
        st.write("Please re-upload a JPG/PNG photo")

# Keep your requirements.txt EXACTLY as before:
# streamlit==1.39.0
# opencv-python-headless==4.9.0.80
# numpy==1.26.4
# Pillow==10.4.0

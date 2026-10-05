import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Ghost Eraser Pro", page_icon="👻", layout="wide")

st.markdown("# 👻 GHOST ERASER - Pro Edition\n**Paint over ghost / person / watermark → Erase in 1 click**")

uploaded = st.file_uploader("Upload Photo", type=["jpg","jpeg","png","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Paint the ghost area (white brush)")
        # Drawable canvas for mask
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=25,
            stroke_color="#FFFFFF",
            background_image=image,
            height=500,
            width=700,
            drawing_mode="freedraw",
            key="canvas",
        )

        method = st.selectbox("Erase Quality", ["NS (Best Quality)", "TELEA (Fast)"])
        blur = st.slider("Edge Clean", 1, 15, 5)
        erase_btn = st.button("✨ ERASE GHOST NOW", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 2. Clean Result")
        result_placeholder = st.empty()
        
        if erase_btn and canvas_result.image_data is not None:
            # Extract mask from canvas
            mask_data = canvas_result.image_data[:, :, 3] # alpha
            if np.max(mask_data) < 10:
                st.warning("Paint over the ghost area first with white brush!")
            else:
                _, binary_mask = cv2.threshold(mask_data.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
                kernel = np.ones((blur, blur), np.uint8)
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
                
                # Resize mask to original image size
                binary_mask = cv2.resize(binary_mask, (w, h))
                
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                flag = cv2.INPAINT_NS if "NS" in method else cv2.INPAINT_TELEA
                inpainted = cv2.inpaint(img_cv, binary_mask, 3, flag)
                
                result_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                result_rgb = cv2.detailEnhance(result_rgb, sigma_s=10, sigma_r=0.15)
                
                result_pil = Image.fromarray(result_rgb)
                result_placeholder.image(result_pil, use_container_width=True)
                st.success(f"Ghost erased! Cleaned {np.count_nonzero(binary_mask)} pixels")
                st.download_button("⬇️ Download HD", result_pil.tobytes(), "ghost_erased.png", "image/png")
        else:
            result_placeholder.info("Paint and click Erase to see result here")

else:
    st.info("👆 Upload a photo to start erasing")
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb", caption="Sample - Try with any photo")

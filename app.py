import cv2
import numpy as np
import gradio as gr
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

# Professional Inpainting Engine
def ghost_erase(image, mask, method="NS", blur=3):
    """
    image: PIL Image
    mask: dict with mask image (from gr.Image with mask)
    method: NS (Navier-Stokes) or TELEA (Fast Marching)
    """
    if image is None:
        return None, "Upload an image first"

    # Handle gradio ImageEditor format
    if isinstance(image, dict):
        bg = image['background']
        mask_img = image['layers'][0] if image['layers'] else None
        if mask_img is None:
            return bg, "Paint over the ghost area to erase"
        img = np.array(bg)
        m = np.array(mask_img)[:, :, 3] # alpha channel is mask
    else:
        # Fallback for older gradio
        img = np.array(image['image']) if isinstance(image, dict) else np.array(image)
        m = np.array(mask) if mask is not None else None
        if m is None:
            return image, "Paint over ghost to erase"

    # Convert mask to binary
    if len(m.shape) == 3:
        m = m[:,:,0]
    _, binary_mask = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)

    # Dilate mask for clean erase
    kernel = np.ones((blur, blur), np.uint8)
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

    # Inpaint - Professional method
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    flag = cv2.INPAINT_NS if method == "NS (Best Quality)" else cv2.INPAINT_TELEA
    inpainted = cv2.inpaint(img_cv, binary_mask, 3, flag)

    result_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    # Post-process - slight sharpening
    result_rgb = cv2.detailEnhance(result_rgb, sigma_s=10, sigma_r=0.15)

    return Image.fromarray(result_rgb), f"✅ Ghost erased! Cleaned {np.count_nonzero(binary_mask)} pixels"

def auto_detect_ghosts(image):
    """Auto detect possible ghosting / shadows using OpenCV"""
    if image is None:
        return None
    img = np.array(image) if not isinstance(image, dict) else np.array(image['background'])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Detect semi-transparent edges (ghosting artifact)
    edges = cv2.Canny(gray, 50, 150)
    # Highlight as suggestion
    overlay = img.copy()
    overlay[edges > 0] = [255, 0, 0] # Red overlay for suggested ghost areas
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return Image.fromarray

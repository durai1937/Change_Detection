import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Function to load images
def load_image(image_file, is_gray=False):
    image = Image.open(image_file)
    image = np.array(image)
    
    if is_gray:
        if len(image.shape) == 3:  # If mistakenly loaded as RGB, convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    return image

# Function to process the images
def process_images(optical_image, sar_image):
    target_size = (256, 256)
    
    optical_image_resized = cv2.resize(np.array(optical_image), target_size)
    
    if len(sar_image.shape) == 2:  # Ensure grayscale image is correctly handled
        sar_image_resized = cv2.resize(sar_image, target_size)
        sar_image_3channel = cv2.merge([sar_image_resized]*3)  # Convert to 3-channel manually
    else:
        sar_image_resized = cv2.resize(sar_image, target_size)
        sar_image_3channel = cv2.cvtColor(sar_image_resized, cv2.COLOR_GRAY2BGR)
    
    return optical_image_resized, sar_image_3channel

# Function to detect changes
def detect_changes(image1, image2):
    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, change_image = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return change_image

# Streamlit UI
st.title("SAR Change Detection App")

optical_image_file = st.file_uploader("Upload Optical Image", type=["jpg", "png"])
sar_image_file = st.file_uploader("Upload SAR Image", type=["jpg", "png"])

if optical_image_file and sar_image_file:
    optical_image = load_image(optical_image_file)
    sar_image = load_image(sar_image_file, is_gray=True)
    optical_resized, sar_resized = process_images(optical_image, sar_image)
    detected_change = detect_changes(optical_resized, sar_resized)
    
    # Display images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(optical_resized, caption="Optical Image", use_column_width=True)
    with col2:
        st.image(sar_resized, caption="SAR Image", use_column_width=True, channels="BGR")
    with col3:
        st.image(detected_change, caption="Detected Changes", use_column_width=True, channels="GRAY")

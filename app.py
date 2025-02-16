import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tkinter as tk
from tkinter import filedialog, messagebox

# Initialize global variables for images
optical_image_normalized = None
sar_image_normalized = None

# Function to load images
def load_image(path, is_gray=False):
    if is_gray:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(path)

# Function to process the images
def process_images(optical_path, sar_path):
    global optical_image_normalized, sar_image_normalized

    # Load and resize images
    optical_image = load_image(optical_path)
    sar_image = load_image(sar_path, is_gray=True)

    # Resize and normalize
    target_size = (256, 256)
    optical_image_resized = cv2.resize(optical_image, target_size)
    sar_image_resized = cv2.resize(sar_image, target_size)
    optical_image_normalized = optical_image_resized / 255.0
    sar_image_normalized = sar_image_resized / np.max(sar_image_resized)
    sar_image_3channel = cv2.cvtColor(sar_image_resized, cv2.COLOR_GRAY2BGR)

    return optical_image_resized, sar_image_3channel

# Function to detect changes
def detect_changes(image1, image2):
    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, change_image = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return change_image

# Function to open optical image file
def open_optical_image():
    global optical_image_path
    optical_image_path = filedialog.askopenfilename(title="Select Optical Image", filetypes=[("Image Files", "*.jpg;*.png")])
    if optical_image_path:
        messagebox.showinfo("Info", "Optical image loaded successfully!")

# Function to open SAR image file
def open_sar_image():
    global sar_image_path
    sar_image_path = filedialog.askopenfilename(title="Select SAR Image", filetypes=[("Image Files", "*.jpg;*.png")])
    if sar_image_path:
        messagebox.showinfo("Info", "SAR image loaded successfully!")

# Function to detect changes and plot results
def detect_and_plot():
    if 'optical_image_path' not in globals() or 'sar_image_path' not in globals():
        messagebox.showerror("Error", "Please load both images before detecting changes.")
        return

    try:
        optical_image, sar_image = process_images(optical_image_path, sar_image_path)
        detected_change = detect_changes(optical_image, sar_image)
        plot_results(optical_image, sar_image, detected_change)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to plot results using Matplotlib
def plot_results(original_optical, original_sar, detected_change):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Optical Image')
    plt.imshow(original_optical)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Original SAR Image')
    plt.imshow(original_sar, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Detected Changes')
    plt.imshow(detected_change, cmap='gray')
    plt.axis('off')
    
    plt.show()

# Tkinter GUI setup
root = tk.Tk()
root.title("SAR Detection App")
root.geometry("300x300")

# Create buttons
browse_optical_button = tk.Button(root, text="Browse Optical Image", command=open_optical_image)
browse_optical_button.pack(pady=10)

browse_sar_button = tk.Button(root, text="Browse SAR Image", command=open_sar_image)
browse_sar_button.pack(pady=10)

detect_button = tk.Button(root, text="Detect Changes", command=detect_and_plot)
detect_button.pack(pady=20)

root.mainloop()

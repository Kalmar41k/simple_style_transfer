"""
Neural Style Transfer App using Streamlit and TensorFlow Hub

This module allows users to apply neural style transfer to images using a pre-trained model from TensorFlow Hub.
Users can upload a content image and a style image, and the model will generate a stylized image.
Additionally, users can swap content and style images to generate a reversed stylization.

Features:
- Upload content and style images via Streamlit UI
- Apply neural style transfer using Google's Magenta model
- Swap content and style images to generate alternative stylized versions
- Download the final stylized image as a high-quality JPEG file

Dependencies:
- TensorFlow
- TensorFlow Hub
- Streamlit
- NumPy
- PIL (Pillow)
- Matplotlib

Author: Kalmar41k
Date: 14.02.2025
"""

import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
from PIL import Image

# print("TF Version: ", tf.__version__)
# print("TF Hub version: ", hub.__version__)
# print("Eager mode enabled: ", tf.executing_eagerly())
# print("GPU available: ", tf.config.list_physical_devices('GPU'))

def crop_center(image):
    """Crops the input image to a square shape from the center."""
    shape = tf.shape(image)
    new_shape = tf.minimum(shape[0], shape[1])
    offset_y = tf.maximum(shape[0] - shape[1], 0) // 2
    offset_x = tf.maximum(shape[1] - shape[0], 0) // 2
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
    return image

def load_image(image_file, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses an image from an uploaded file."""
    img = tf.io.decode_image(image_file.read(), channels=3, dtype=tf.float32) # Decode image to TensorFlow tensor
    img = crop_center(img) # Crop the image to a square
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio) # Resize the image
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def show_n(images, titles=('',)):
    """Displays multiple images in a row using Streamlit."""
    cols = st.columns(len(images))  # Create columns based on the number of images
    for img, title, col in zip(images, titles, cols):
        img = np.array(img[0].numpy() * 255, dtype=np.uint8)  # Convert tensor to NumPy array
        img = Image.fromarray(img)
        with col:
            st.image(img, caption=title, use_column_width=True)

# Streamlit UI setup
st.title("Neural Style Transfer")
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])
output_image_size = 384 # Default output image size

# Load TensorFlow Hub model for style transfer
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Initialize session state variables
if 'content_image' not in st.session_state:
    st.session_state.content_image = None
if 'style_image' not in st.session_state:
    st.session_state.style_image = None
if 'show_swap_button' not in st.session_state:
    st.session_state.show_swap_button = False
if 'stylized_image' not in st.session_state:
    st.session_state.stylized_image = None

if content_file is None:
    st.session_state.content_image = None
if style_file is None:
    st.session_state.style_image = None

# Process uploaded images
if content_file and style_file:
    if st.button("Apply Style Transfer"):
        # Show spinner during the model loading and style transfer process
        with st.spinner("Model is loading and applying style transfer..."):
            content_img_size = (output_image_size, output_image_size)
            style_img_size = (256, 256)
            content_image = load_image(content_file, content_img_size)
            style_image = load_image(style_file, style_img_size)

            # Apply slight blurring to the style image for better blending
            style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

            # Run the neural style transfer model
            outputs = hub_module(content_image, style_image)
            stylized_image = outputs[0]

        # Store the images in session state
        st.session_state.content_image = content_image
        st.session_state.style_image = style_image
        st.session_state.stylized_image = stylized_image

        # Display the images
        show_n([content_image, style_image, stylized_image], 
               titles=['Content image', 'Style image', 'Stylized image'])

        # Enable the "Swap Images" button
        st.session_state.show_swap_button = True

# Swap images feature
if st.session_state.content_image is not None and st.session_state.style_image is not None \
    and content_file and style_file and st.session_state.show_swap_button:
    if st.button("Swap Images"):
        with st.spinner("Applying swapped style transfer..."):

            # Swap content and style images
            swapped_content_image = st.session_state.style_image
            swapped_style_image = st.session_state.content_image

            # Apply style transfer again with swapped roles
            outputs = hub_module(swapped_content_image, swapped_style_image)
            swapped_stylized_image = outputs[0]

        # Update session state with swapped images
        st.session_state.content_image = swapped_content_image
        st.session_state.style_image = swapped_style_image
        st.session_state.stylized_image = swapped_stylized_image

        # Display swapped results
        show_n([swapped_content_image, swapped_style_image, swapped_stylized_image],
               titles=['Content image', 'Style image', 'Stylized image'])

# Download button for the final stylized image
if 'stylized_image' in st.session_state and st.session_state.stylized_image is not None:
    # Convert TensorFlow tensor to PIL image
    img_array = np.array(st.session_state.stylized_image[0].numpy() * 255, dtype=np.uint8)
    img_pil = Image.fromarray(img_array)

    # Resize the final image for better quality
    img_pil = img_pil.resize((1024, 1024), Image.LANCZOS)

    # Save as JPEG format
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="JPEG", quality=95)   # Set quality to 95%
    img_bytes = img_bytes.getvalue()

    # Provide download button
    st.download_button(
        label="Download Stylized Image",
        data=img_bytes,
        file_name="stylized_image.jpg",
        mime="image/jpeg"
    )

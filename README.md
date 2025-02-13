# Neural Style Transfer App using Streamlit and TensorFlow Hub

This project allows users to apply neural style transfer to images using a pre-trained model from TensorFlow Hub. The app provides a simple UI built with Streamlit where users can upload a content image and a style image. The model then generates a stylized version of the content image based on the style image. Additionally, users can swap the content and style images to generate a reversed stylization.

## Features

- Upload content and style images via Streamlit UI
- Apply neural style transfer using Google's Magenta model
- Swap content and style images to generate alternative stylized versions
- Download the final stylized image as a high-quality JPEG file

## Dependencies

The following dependencies are required to run the project:

- TensorFlow
- TensorFlow Hub
- Streamlit
- NumPy
- Pillow (PIL)
- Matplotlib

To install them, use the following command:
conda create --name <env> --file environment.yml

## How to Run
Clone this repository:
git clone https://github.com/Kalmar41k/simple_style_transfer.git
cd neural-style-transfer-app
conda create --name nst-env --file environment.yml
streamlit run app.py

Open the app in your browser and upload your images to see the style transfer in action.

## Author
Kalmar41k

## Date
14.02.2025
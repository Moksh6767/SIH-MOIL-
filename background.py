import streamlit as st
import base64
import os
import pandas as pd
import numpy as np


def set_background(image_path):
    """
    Reads a local image, converts it to base64, and injects it as the Streamlit app background.
    """
    with open(image_path, "rb") as f:
        img_data = f.read()

    b64_encoded = base64.b64encode(img_data).decode()

    # Injecting the CSS
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:background/jpeg;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# CALL THE FUNCTION HERE (Replace with your actual image file name)
set_background("background.jpg")

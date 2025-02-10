import streamlit as st 
import google.generativeai as genai
import json
from streamlit_lottie import st_lottie
import requests

# MUST be the very first Streamlit command


# Now you can add your custom CSS and other content
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #ff512f, #dd2476);
            padding: 20px;
            border-radius: 15px;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #6a0572;
            color: white;
            border-radius: 10px;
            padding: 8px;
            font-weight: bold;
        }
        .stTextArea>div>textarea {
            background: #ff9966;
            color: black;
            border-radius: 15px;
            padding: 12px;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(to right, #12c2e9, #c471ed, #f64f59);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Continue with the rest of your app...
# Sidebar Configuration, Lottie animation, chatbot interaction, etc.

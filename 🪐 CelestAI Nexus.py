import streamlit as st 
import google.generativeai as genai
import json
from streamlit_lottie import st_lottie
import requests

# Configure the page

# Custom CSS for a futuristic, extraterrestrial theme
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #000428, #004e92);
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
            padding: 20px;
            border-radius: 15px;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #162447;
            color: white;
            border-radius: 10px;
            padding: 8px;
            font-weight: bold;
        }
        .stTextArea>div>textarea {
            background: #1f4068;
            color: white;
            border-radius: 15px;
            padding: 12px;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(to right, #11998e, #38ef7d);
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

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🛸 Welcome to Alien Signal Classification App")
    api_key = st.text_input("🔑 Enter Google Gemini API Key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)
    
    st.markdown("#### 🛸 Example Queries:")
    example_prompts = [
        "How can we classify alien signals?",
        "What is the WOW! signal?",
        "How does AI help in detecting extraterrestrial communications?",
        "What are the key frequencies in deep space signals?",
    ]
    for prompt in example_prompts:
        if st.button(prompt):
            st.session_state.user_input = prompt

    st.markdown("---")

# Function to load Lottie animation
def load_lottie_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

# Load and display Lottie animation
animation_path = "AI.json"  # Update with the correct path
try:
    animation = load_lottie_file(animation_path)
    st_lottie(animation, speed=1, loop=True, quality="high", height=250, key="animation")
except Exception as e:
    st.error(f"Error loading animation: {e}")

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <h1 style="color: #00ffcc; text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.5);"><strong>🛸 Alien Signal Classification App 📡</strong></h1>
        <p style="color: #b3ffab; font-size: 20px; font-weight: bold;">Analyze mysterious cosmic signals and classify them using AI! 👽</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Chatbot Interaction for Alien Signal Analysis
user_input = st.text_area("💬 Ask a question about alien signals:")
if st.button("🛸 Analyze Signal"):
    if not api_key:
        st.warning("⚠️ Please enter a valid Google Gemini API key in the sidebar.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(user_input)
            st.success("✨ AI Analysis:")
            st.markdown(f"<p style='background: linear-gradient(to right, #3a1c71, #d76d77, #ffaf7b); padding:12px; border-radius:12px; color:white; font-weight:bold;'>{response.text}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

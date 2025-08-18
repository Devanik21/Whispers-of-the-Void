import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from scipy import stats


# Set page configuration with an attractive layout

# Custom CSS for a modern, advanced design
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(135deg, #c9eaff, #ffffff);
        color: black;
        font-family: 'Helvetica', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #dde9f0;
        border-right: 2px solid #a7c6db;
        color: black;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #9cd4f0;
        font-weight: 700;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #005f73;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0a9396;
        transform: scale(1.05);
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>div>input, .stMultiSelect>div {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 10px;
    }
    .highlight-cell {
        animation: highlight 1s ease;
    }
    @keyframes highlight {
        0% { background-color: #dff9fb; }
        100% { background-color: transparent; }
    }
    </style>
    """, unsafe_allow_html=True)

# Title with a modern look and emoji
st.title("📊 Advanced Data Analysis Dashboard")

# Introductory text with cleaner Markdown formatting
st.markdown("""
    ### Welcome to the **Advanced Data Analysis** section! 🎉
    Explore the following features:
    - 📊 **Statistical Analysis**: Gain insights from summary statistics.
    - 🔗 **Correlation Analysis**: Visualize relationships between features.
    - 🔍 **Feature Distribution**: Explore data distributions and custom visualizations.
    - 🛠️ **Custom Analysis Tool**: Execute custom Pandas code for deeper insights.
    ---
""")

# Sidebar section for file upload with emoji and smooth interaction
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file to begin your analysis", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data and display it with animations
        data = pd.read_csv(uploaded_file)
        st.subheader("🔍 Dataset Overview")
        st.write(f"📏 **Rows and Columns**: {data.shape[0]} rows, {data.shape[1]} columns")
        st.write("🔎 **Data Preview:**")
        st.dataframe(data.head(10))  # Show the first 10 rows

        # Add cool summary statistics with better styling and emojis
        if st.checkbox("📊 Show Statistical Summary", value=True):
            styled_df = data.describe().T.style.format("{:.2f}").background_gradient(cmap="coolwarm")
            st.write(styled_df)
        
        # Filter numeric columns for correlation heatmap
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if numeric_columns:
            st.subheader("📈 Correlation Heatmap 🔗")
            selected_columns = st.multiselect("🎯 Select numeric features for correlation analysis", numeric_columns, default=numeric_columns)

            if len(selected_columns) > 1:
                corr = data[selected_columns].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask to show only one triangle of the heatmap
                
                # Plot correlation heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', mask=mask, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.75})
                ax.set_title("💡 Correlation Heatmap", fontsize=18)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Please select at least two numeric features for correlation.")
        else:
            st.warning("⚠️ No numeric columns available for correlation analysis.")
        
        # Add interactive distribution analysis with smooth visuals
        st.subheader("🔬 Explore Feature Distribution")
        selected_feature = st.selectbox("📊 Select a feature for analysis", data.columns)

        if pd.api.types.is_numeric_dtype(data[selected_feature]):
            fig, ax = plt.subplots()
            sns.histplot(data[selected_feature], kde=True, ax=ax, color="#0077b6")
            ax.set_title(f"📊 Distribution of {selected_feature}", fontsize=15)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.countplot(x=data[selected_feature], ax=ax, palette="Set2")
            ax.set_title(f"📊 Count Plot of {selected_feature}", fontsize=15)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Allow custom Pandas code execution with enhanced styling
        st.sidebar.subheader("📝 Custom Analysis Tool")
        custom_code = st.sidebar.text_area("✍️ Write your custom Pandas code:", "data.head()")

        st.subheader("🛠️ Custom Code Output")
        try:
            result = eval(custom_code)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)  # Display DataFrame with better UI
            else:
                st.write(result)
        except Exception as e:
            st.error(f"❌ Error in custom code: {e}")

    except Exception as e:
        st.error(f"❌ Error loading the dataset: {e}")
else:
    st.info("📂 Please upload a CSV dataset to get started.")

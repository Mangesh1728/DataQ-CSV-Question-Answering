import streamlit as st
import pandas as pd
from transformers import pipeline

# Set page title and background color
st.set_page_config(page_title="DataQ: CSV Question Answering", layout="wide", page_icon="🌐")
 # Add custom CSS for background color
st.markdown(
        """
        <style>
        .stApp {
            background-color: #ABF7EA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Header and subheader
st.title("DataQ: CSV Question Answering")
st.subheader("Upload a CSV file and ask questions about the data.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV file
    table = pd.read_csv(uploaded_file)
    table = table.astype(str)

    # Display uploaded data
    st.write("Uploaded Data:")
    st.write(table)

    # Query input
    query = st.text_input("Enter your query:", "Who has scored the highest runs?")

    # Task-specific pipeline
    tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

    # Button to trigger query
    if st.button("Get Answer"):
        # Perform query and display result
        answer = tqa(table=table, query=query)["answer"]
        st.success(f"Answer: {answer}")

else:
    st.warning("Please upload a CSV file.")
# Display a disclaimer or additional information at the bottom
# Display disclaimer using Markdown
# Add description in the sidebar
st.sidebar.title(" 👇DESCRIPTION 👇")
st.sidebar.markdown("""
⚠️ Disclaimer:
1. The answers provided by this app are based on the CSV File data uploaded by the user.
2. The accuracy and reliability of the answers may vary depending on the quality and completeness of the data.
3. This app is intended for educational and informational purposes only.
4. Users should verify critical information independently before making decisions based on the app's output.
""")

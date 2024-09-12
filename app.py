import streamlit as st
import json
import pandas as pd
from utils import load_json, validate_json
from evaluation import evaluate_rag
from config import OPENAI_MODELS

st.set_page_config(layout="wide", page_title="RAG Evaluation App")


def main():
    st.title("RAG Evaluation App")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model = st.selectbox("Select OpenAI Model", options=OPENAI_MODELS)

    # Main content
    uploaded_file = st.file_uploader("Upload JSON file", type="json")

    if uploaded_file is not None:
        try:
            data = load_json(uploaded_file)
            if validate_json(data):
                st.success("JSON file uploaded successfully!")
                display_json_data(data)

                if st.button("Evaluate"):
                    if api_key:
                        with st.spinner("Evaluating..."):
                            results = evaluate_rag(data, api_key, model)
                        display_results(results)
                    else:
                        st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                st.error("Invalid JSON structure. Please upload a valid file.")
        except json.JSONDecodeError:
            st.error("Error decoding JSON. Please upload a valid JSON file.")


def display_json_data(data):
    st.subheader("Uploaded JSON Data")
    df = pd.json_normalize(data)
    st.dataframe(df)


def display_results(results):
    st.subheader("Evaluation Results")
    for metric, result in results.items():
        st.metric(label=metric, value=f"{result['score']:.2f}")
        st.text(f"Reason: {result['reason']}")


if __name__ == "__main__":
    main()

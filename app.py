import streamlit as st
import json
import pandas as pd
import base64
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
                            results = evaluate_rag_with_progress(data, api_key, model)
                        display_results_table(results)
                        provide_download_button(results)
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


def evaluate_rag_with_progress(data, api_key, model):
    results_list = []
    metrics = [
        "Faithfulness",
        "Context Precision",
        "Relevance",
        "Context Recall",
        "Context Relevancy",
        "Context Entities Recall",
        "Answer Semantic Similarity",
        "Answer Correctness",
    ]

    status_text = st.empty()

    status_text.text("Evaluating...")
    results = evaluate_rag(data, api_key, model)

    for metric, result in results.items():
        results_list.append(
            {"metric": metric, "score": result["score"], "reason": result["reason"]}
        )

    status_text.text("Evaluation complete!")
    return results_list


def display_results_table(results):
    st.subheader("Evaluation Results")

    df = pd.DataFrame(
        [
            (result["metric"], result["score"], result["reason"])
            for result in results
        ],
        columns=["Metric", "Score", "Reason"],
    )
    st.table(df)


def provide_download_button(results):
    df = pd.DataFrame(
        [
            (result["metric"], result["score"], result["reason"])
            for result in results
        ],
        columns=["Metric", "Score", "Reason"]
    )
    
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="evaluation_results.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()

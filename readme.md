# RAG Evaluation App

This Streamlit application allows users to evaluate Retrieval-Augmented Generation (RAG) systems by uploading JSON files containing query and response data. The app uses OpenAI's API to perform various evaluations on the RAG system's performance.

## Features

- Upload and validate JSON files containing RAG system data
- Configure OpenAI API key and model selection
- Evaluate RAG system performance across multiple metrics
- Display evaluation results in a table format
- Download evaluation results as a CSV file

## Prerequisites

- Python 3.7+
- Streamlit
- OpenAI API key

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rag-evaluation-app.git
   cd rag-evaluation-app
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. In the sidebar, enter your OpenAI API key and select the desired model.

4. Upload a JSON file containing your RAG system data using the file uploader.

5. If the JSON file is valid, click the "Evaluate" button to start the evaluation process.

6. View the evaluation results in the table displayed on the page.

7. Download the results as a CSV file using the "Download Results CSV" button.

## JSON File Format

The uploaded JSON file should follow this structure:

see `output.json` in the repo

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "statsmodels",
#     "scikit-learn",
#     "missingno",
#     "python-dotenv",
#     "requests",
#     "seaborn",
# ]
# ///
import os
import sys
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from dotenv import load_dotenv
from PIL import Image
import chardet
from io import BytesIO
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Fetch AI Proxy token from .env file
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    logging.error("AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

# Define headers for API request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIPROXY_TOKEN}'
}

# Function to request AI to generate the narrative story with timeout handling
def get_ai_story(dataset_summary, dataset_info, visualizations):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    prompt = f"""
    Below is a detailed summary and analysis of a dataset. Please generate a **rich and engaging narrative** about this dataset analysis, including:

    1. **The Data Received**: Describe the dataset vividly. What does the data represent? What are its features? What is the significance of this data? Create a compelling story around it.
    2. **The Analysis Carried Out**: Explain the analysis methods used. Highlight techniques like missing value handling, outlier detection, clustering, and dimensionality reduction (PCA). How do these methods provide insights?
    3. **Key Insights and Discoveries**: What were the major findings? What trends or patterns emerged that can be interpreted as discoveries? Were there any unexpected results?
    4. **Implications and Actions**: Discuss the implications of these findings. How do they influence decisions? What actionable recommendations would you provide based on the analysis?
    5. **Visualizations**: Describe the visualizations included. What do they reveal about the data? How do they complement the analysis and findings?

    **Dataset Summary**:
    {dataset_summary}

    **Dataset Info**:
    {dataset_info}

    **Visualizations**:
    {visualizations}
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)  # Set a 30-second timeout
        response.raise_for_status()  # Will raise HTTPError for bad responses
    except requests.exceptions.Timeout:
        logging.error("Request timed out. AI service is too slow.")
        return "Error: Timeout occurred during narrative generation."
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return "Error: Unable to generate narrative. Please check the AI service."

    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No narrative generated.")

# Function to load dataset with automatic encoding detection
def load_data(file_path, chunk_size=50000):
    try:
        data = pd.read_csv(file_path, encoding='utf-8', chunksize=chunk_size)  # Use chunksize for large files
        return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# Perform basic analysis
def basic_analysis(data):
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Outlier detection using IQR
def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

# Function to process data in chunks concurrently
def process_chunks_in_parallel(file_path):
    data_chunks = load_data(file_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda chunk: process_data_chunk(chunk), data_chunks))
    return results

# Function to process each chunk of data
def process_data_chunk(chunk):
    # Perform all necessary operations for each chunk
    analysis = basic_analysis(chunk)
    outliers = outlier_detection(chunk)
    combined_analysis = {**analysis, **outliers}
    return combined_analysis

# Enhanced plot saving with clearer visualization
def save_plot(fig, plot_name):
    plot_path = f"{plot_name}.png"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved as {plot_path}")
    return plot_path

# Enhanced Correlation heatmap
def generate_correlation_matrix(data):
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        logging.warning("No numeric columns for correlation matrix.")
        return None
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix", fontsize=16)
    return save_plot(fig, "correlation_matrix")

# Full analysis workflow with enhanced structure
def analyze_and_generate_output(file_path):
    start_time = time.time()
    data_chunks = load_data(file_path)

    # Process chunks concurrently
    analysis_results = process_chunks_in_parallel(file_path)

    # Generate visualizations for the entire dataset (or summary if chunked)
    image_paths = {
        'correlation_matrix': generate_correlation_matrix(data_chunks),
    }

    data_info = {
        "filename": file_path,
        "summary": analysis_results[0]['summary'],  # Assuming summary from first chunk for demonstration
        "missing_values": analysis_results[0]['missing_values'],
        "outliers": analysis_results[0]['outliers']
    }

    narrative = get_ai_story(data_info["summary"], data_info["missing_values"], image_paths)
    if not narrative:
        narrative = "Error: Narrative generation failed. Please verify the AI service."

    # Save the results
    save_readme(f"Dataset Analysis: {narrative}")
    
    end_time = time.time()
    logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds.")
    return narrative, image_paths

# Main entry point
def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()




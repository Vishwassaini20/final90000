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
import chardet  # To detect file encoding
from io import BytesIO
import argparse
import logging

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

# Function to request AI to generate the narrative story
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Will raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return "Error: Unable to generate narrative. Please check the AI service."

    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No narrative generated.")

# Function to load dataset with automatic encoding detection
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect the encoding of the file
        encoding = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"Data loaded with {encoding} encoding.")
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

# Enhanced PCA Visualization
def generate_pca_plot(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.shape[1] < 2:
        logging.warning("Insufficient numeric columns for PCA.")
        return None
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], ax=ax, palette="viridis")
    ax.set_title("PCA Plot", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=14)
    ax.set_ylabel("Principal Component 2", fontsize=14)
    return save_plot(fig, "pca_plot")

# Enhanced DBSCAN clustering with better color palette
def dbscan_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.empty:
        logging.warning("No numeric data for DBSCAN.")
        return None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=numeric_data['cluster'], palette="coolwarm", ax=ax)
    ax.set_title("DBSCAN Clustering", fontsize=16)
    return save_plot(fig, "dbscan_clusters")

# Hierarchical clustering with clearer labels
def hierarchical_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.empty:
        logging.warning("No numeric data for hierarchical clustering.")
        return None
    linked = linkage(numeric_data, 'ward')
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linked, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=16)
    return save_plot(fig, "hierarchical_clustering")

# Save README file with detailed structure
def save_readme(content):
    try:
        readme_path = "README.md"
        with open(readme_path, "w") as f:
            f.write(content)
        logging.info(f"README saved in the current directory.")
    except Exception as e:
        logging.error(f"Error saving README: {e}")
        sys.exit(1)

# Full analysis workflow with enhanced structure
def analyze_and_generate_output(file_path):
    data = load_data(file_path)
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}

    image_paths = {
        'correlation_matrix': generate_correlation_matrix(data),
        'pca_plot': generate_pca_plot(data),
        'dbscan_clusters': dbscan_clustering(data),
        'hierarchical_clustering': hierarchical_clustering(data)
    }

    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }

    narrative = get_ai_story(data_info["summary"], data_info["missing_values"], image_paths)
    if not narrative:
        narrative = "Error: Narrative generation failed. Please verify the AI service."
    save_readme(f"Dataset Analysis: {narrative}")
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

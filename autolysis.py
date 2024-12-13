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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import chardet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from dotenv import load_dotenv
from scipy import stats
from PIL import Image
import missingno as msno
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

# Function to detect file encoding
def detect_encoding(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Function to load and clean the dataset
def load_and_clean_data(filename):
    encoding = detect_encoding(filename)
    df = pd.read_csv(filename, encoding=encoding)
    
    # Drop rows with all NaN values
    df.dropna(axis=0, how='all', inplace=True)
    
    # Fill missing values in numeric columns with the mean of the column
    numeric_columns = df.select_dtypes(include='number')
    df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())
    
    # Handle missing values in non-numeric columns (e.g., fill with 'Unknown')
    non_numeric_columns = df.select_dtypes(exclude='number')
    df[non_numeric_columns.columns] = non_numeric_columns.fillna('Unknown')
    
    return df

# Function to summarize the dataset
def summarize_data(df):
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'types': df.dtypes.to_dict(),
        'descriptive_statistics': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary

# Outlier detection function using Z-Score
def detect_outliers(df):
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_df))
    outliers = (z_scores > 3).sum(axis=0)
    outlier_info = {
        column: int(count) for column, count in zip(numeric_df.columns, outliers)
    }
    return outlier_info

# Correlation analysis function
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    return correlation_matrix.to_dict()

# Cluster analysis using DBSCAN
def dbscan_clustering(df):
    numeric_data = df.select_dtypes(include=np.number).dropna()
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

# PCA for dimensionality reduction (optional)
def perform_pca(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]
    return df

# Function to save visualizations
def save_plot(fig, plot_name):
    plot_path = f"{plot_name}.png"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved as {plot_path}")
    return plot_path

# Create visualizations for missing data, correlation, and PCA
def create_visualizations(df):
    # Visualization for missing data
    msno.matrix(df)
    plt.tight_layout()
    missing_img = 'missing_data.png'
    plt.savefig(missing_img)
    plt.close()

    # Filter numeric columns for correlation heatmap
    numeric_df = df.select_dtypes(include='number')
    
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        correlation_img = 'correlation_matrix.png'
        plt.tight_layout()
        plt.savefig(correlation_img)
        plt.close()
    else:
        correlation_img = None

    # PCA scatter plot
    perform_pca(df)
    pca_img = 'pca_plot.png'
    plt.scatter(df['PCA1'], df['PCA2'])
    plt.title('PCA Plot')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig(pca_img)
    plt.close()

    return {'missing_img': missing_img, 'correlation_img': correlation_img, 'pca_img': pca_img}

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
    data = load_and_clean_data(file_path)
    analysis = summarize_data(data)
    outliers = detect_outliers(data)
    combined_analysis = {**analysis, **outliers}

    image_paths = create_visualizations(data)

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

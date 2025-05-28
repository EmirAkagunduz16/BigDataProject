"""
Flask API server for the Academic Paper Clustering application.
This API connects the React frontend to the PySpark backend.
"""

import os
import sys
import json
import uuid
import time
import threading
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.arxiv_data_collector import ArXivDataCollector
from src.spark_clustering import SparkTextClustering

# Flask app setup
app = Flask(__name__, 
           static_folder='../frontend/build', 
           static_url_path='/')
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Enable CORS for all routes
CORS(app, origins=["http://localhost:3000", "http://localhost:5000", "http://127.0.0.1:5000"])

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, '..', 'visualizations')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# In-memory job tracking
active_jobs = {}

def get_job_status(job_id):
    """Get the status of a job"""
    if job_id not in active_jobs:
        return {"status": "not_found"}
    return active_jobs[job_id]

def update_job_status(job_id, status, progress=0, message="", result=None):
    """Update the status of a job"""
    active_jobs[job_id] = {
        "status": status,
        "progressPercent": progress,
        "message": message,
        "result": result,
        "updatedAt": datetime.now().isoformat()
    }

# API Routes
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get project statistics"""
    try:
        # Check if we have clustered data
        clustered_file = os.path.join(DATA_DIR, 'clustered_papers.csv')
        if os.path.exists(clustered_file):
            df = pd.read_csv(clustered_file)
            stats = {
                "totalPapers": len(df),
                "totalClusters": df['cluster'].nunique(),
                "categoriesCount": df['primary_category'].nunique(),
                "clusterSizes": df['cluster'].value_counts().to_dict(),
                "processingStatus": "completed",
                "lastRun": datetime.fromtimestamp(os.path.getmtime(clustered_file)).strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            # Check if we have raw data
            raw_file = os.path.join(DATA_DIR, 'arxiv_papers.csv')
            if os.path.exists(raw_file):
                df = pd.read_csv(raw_file)
                stats = {
                    "totalPapers": len(df),
                    "totalClusters": 0,
                    "categoriesCount": df['primary_category'].nunique(),
                    "clusterSizes": {},
                    "processingStatus": "data_collected",
                    "lastRun": datetime.fromtimestamp(os.path.getmtime(raw_file)).strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                stats = {
                    "totalPapers": 0,
                    "totalClusters": 0,
                    "categoriesCount": 0,
                    "clusterSizes": {},
                    "processingStatus": "not_started",
                    "lastRun": ""
                }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/collect-data', methods=['POST'])
def collect_data():
    """Start data collection process"""
    try:
        data = request.json
        categories = data.get('categories', [])
        max_results = data.get('maxResults', 1000)
        use_primary_only = data.get('usePrimaryOnly', False)
        
        # Validate input
        if not categories:
            return jsonify({"error": "No categories provided"}), 400
        
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        update_job_status(job_id, "started", 0, "Initializing data collection...")
        
        # Start collection in a background thread
        def run_collection():
            try:
                update_job_status(job_id, "running", 10, "Setting up ArXiv collector...")
                
                # Initialize collector
                collector = ArXivDataCollector(max_results=max_results, delay=0.5)
                
                if use_primary_only:
                    update_job_status(job_id, "running", 20, f"Collecting papers with primary category filter from: {', '.join(categories)}")
                    # Use new method
                    df = collector.collect_papers_by_primary_category(categories)
                else:
                    update_job_status(job_id, "running", 20, f"Collecting papers from categories: {', '.join(categories)}")
                    # Collect papers
                    df = collector.collect_papers_by_category(categories)
                    
                    update_job_status(job_id, "running", 50, "Filtering by primary categories...")
                    # Filter by primary categories
                    original_count = len(df)
                    df = df[df['primary_category'].isin(categories)]
                    filtered_count = len(df)
                    
                    update_job_status(job_id, "running", 60, f"Filtered {original_count} -> {filtered_count} papers")
                
                update_job_status(job_id, "running", 70, "Preprocessing data...")
                
                # Preprocess
                df_clean = collector.preprocess_dataframe(df)
                
                update_job_status(job_id, "running", 90, "Saving data...")
                
                # Save data
                output_file = os.path.join(DATA_DIR, 'arxiv_papers.csv')
                collector.save_data(df_clean, output_file)
                
                # Generate statistics
                stats = {
                    'total_papers': len(df_clean),
                    'unique_categories': df_clean['primary_category'].nunique(),
                    'category_distribution': df_clean['primary_category'].value_counts().to_dict(),
                    'missing_categories': list(set(categories) - set(df_clean['primary_category'].unique()))
                }
                
                update_job_status(job_id, "completed", 100, 
                    f"Successfully collected {len(df_clean)} papers from {df_clean['primary_category'].nunique()} categories")
                
                # Store stats in job status
                job_statuses[job_id]['stats'] = stats
                
            except Exception as e:
                update_job_status(job_id, "failed", 0, f"Error during collection: {str(e)}")
        
        # Start background thread
        thread = threading.Thread(target=run_collection)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "jobId": job_id,
            "message": "Data collection started",
            "usePrimaryOnly": use_primary_only
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cluster', methods=['POST'])
def perform_clustering():
    """Start clustering process"""
    try:
        data = request.json
        vocab_size = data.get('vocabSize', 5000)
        k_range = data.get('kRange')
        specific_k = data.get('specificK')
        
        # Validate input
        data_file = os.path.join(DATA_DIR, 'arxiv_papers.csv')
        if not os.path.exists(data_file):
            return jsonify({"error": "No data available. Please collect data first."}), 400
        
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        update_job_status(job_id, "started", 0, "Initializing clustering...")
        
        # Start clustering in a background thread
        def run_clustering():
            try:
                # Set up k range
                if k_range:
                    k_min, k_max = k_range
                    k_range_list = range(k_min, k_max + 1)
                else:
                    k_range_list = None
                
                update_job_status(job_id, "running", 10, "Setting up Spark session...")
                
                # Initialize clustering
                clustering = SparkTextClustering()
                
                try:
                    # Data loading
                    update_job_status(job_id, "running", 20, "Loading data...")
                    df = clustering.load_data(data_file)
                    
                    # Text preprocessing
                    update_job_status(job_id, "running", 30, "Preprocessing text...")
                    df = clustering.preprocess_text(['title', 'summary'])
                    
                    # Feature extraction
                    update_job_status(job_id, "running", 40, "Extracting TF-IDF features...")
                    df_features = clustering.create_features(vocab_size=vocab_size, min_df=2)
                    
                    # Find optimal k or use specific k
                    if k_range_list:
                        update_job_status(job_id, "running", 50, f"Finding optimal k in range {k_range}...")
                        optimal_k, costs, silhouette_scores = clustering.find_optimal_k(
                            k_range=k_range_list, 
                            iterations=50
                        )
                    else:
                        optimal_k = specific_k
                    
                    # Perform clustering
                    update_job_status(job_id, "running", 70, f"Performing K-means clustering with k={optimal_k}...")
                    df_clustered = clustering.perform_clustering(
                        k=optimal_k, 
                        max_iterations=100
                    )
                    
                    # Analyze clusters
                    update_job_status(job_id, "running", 80, "Analyzing clusters...")
                    cluster_analysis = clustering.analyze_clusters(top_words=15)
                    
                    # Create visualizations
                    update_job_status(job_id, "running", 90, "Creating visualizations...")
                    clustering.create_visualizations()
                    
                    # Save results
                    update_job_status(job_id, "running", 95, "Saving results...")
                    result_df = clustering.save_results()
                    
                    # Update job status with results
                    update_job_status(
                        job_id, 
                        "completed", 
                        100, 
                        f"Clustering completed successfully with k={optimal_k}",
                        {
                            "optimalK": optimal_k,
                            "silhouetteScore": clustering.silhouette_score,
                            "clusterSizes": {k: v['size'] for k, v in cluster_analysis.items()}
                        }
                    )
                finally:
                    # Always stop the Spark session
                    clustering.stop_spark()
                    
            except Exception as e:
                update_job_status(job_id, "failed", 0, f"Error: {str(e)}")
        
        # Start the thread
        threading.Thread(target=run_clustering).start()
        
        return jsonify({"jobId": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/job-status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get the status of a job"""
    status = get_job_status(job_id)
    if status.get("status") == "not_found":
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)

@app.route('/api/clustered-papers', methods=['GET'])
def get_clustered_papers():
    """Get the clustered papers"""
    try:
        clustered_file = os.path.join(DATA_DIR, 'clustered_papers.csv')
        if not os.path.exists(clustered_file):
            return jsonify({"error": "No clustered data available"}), 404
        
        df = pd.read_csv(clustered_file)
        
        # Convert to list of dictionaries
        papers = df.to_dict(orient='records')
        
        return jsonify({"papers": papers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    """Get the visualization URLs"""
    try:
        # Check if visualizations exist and build URLs
        visualizations = {}
        
        # Check each visualization file
        files_to_check = {
            "clusterSizesUrl": "cluster_sizes.html",
            "categoryDistributionUrl": "category_distribution.png", 
            "clusterCategoryHeatmapUrl": "cluster_category_heatmap.png",
            "wordcloudsUrl": "cluster_wordclouds.png"
        }
        
        for url_key, filename in files_to_check.items():
            file_path = os.path.join(VISUALIZATIONS_DIR, filename)
            if os.path.exists(file_path):
                visualizations[url_key] = f"/visualizations/{filename}"
            else:
                visualizations[url_key] = None
        
        # Also return file existence status
        visualizations["filesExist"] = {
            key: visualizations[key] is not None for key in files_to_check.keys()
        }
        
        return jsonify(visualizations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualizations/<path:filename>')
def serve_visualizations(filename):
    """Serve visualization files"""
    try:
        file_path = os.path.join(VISUALIZATIONS_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Add CORS headers for iframe support
        response = send_from_directory(VISUALIZATIONS_DIR, filename)
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['Content-Security-Policy'] = "frame-ancestors 'self'"
        
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-visualizations', methods=['GET'])
def download_visualizations():
    """Download all visualizations as a ZIP file"""
    # This would normally create a ZIP file of all visualizations
    # For simplicity, we'll just serve one of the visualization files
    try:
        cluster_sizes_html = os.path.join(VISUALIZATIONS_DIR, 'cluster_sizes.html')
        if os.path.exists(cluster_sizes_html):
            return send_file(cluster_sizes_html, as_attachment=True)
        else:
            return jsonify({"error": "Visualizations not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve the React frontend"""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
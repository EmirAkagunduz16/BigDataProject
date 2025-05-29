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
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
try:
    from src.arxiv_data_collector import ArXivDataCollector
    from src.spark_clustering import SparkTextClustering
    print("✅ Successfully imported project modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    traceback.print_exc()

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
JOBS_DIR = os.path.join(BASE_DIR, '..', 'jobs')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory job tracking with persistent storage
active_jobs = {}

def load_job_status(job_id):
    """Load job status from file"""
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(job_file):
        try:
            with open(job_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading job {job_id}: {e}")
    return None

def save_job_status(job_id, status_data):
    """Save job status to file"""
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    try:
        with open(job_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        print(f"Error saving job {job_id}: {e}")

def get_job_status(job_id):
    """Get the status of a job"""
    # First check in-memory storage
    if job_id in active_jobs:
        return active_jobs[job_id]
    
    # Then check persistent storage
    status = load_job_status(job_id)
    if status:
        # Load back into memory
        active_jobs[job_id] = status
        return status
    
    return {"status": "not_found", "error": "Job ID not found"}

def update_job_status(job_id, status, progress=0, message="", result=None):
    """Update the status of a job"""
    status_data = {
        "status": status,
        "progressPercent": progress,
        "message": message,
        "result": result,
        "updatedAt": datetime.now().isoformat()
    }
    
    # Update in-memory
    active_jobs[job_id] = status_data
    
    # Save to persistent storage
    save_job_status(job_id, status_data)
    
    print(f"Job {job_id}: {status} - {progress}% - {message}")

# API Routes
@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to verify API is working"""
    try:
        return jsonify({
            "status": "ok",
            "message": "API is working",
            "data_dir": DATA_DIR,
            "data_file_exists": os.path.exists(os.path.join(DATA_DIR, 'arxiv_papers.csv')),
            "clustered_file_exists": os.path.exists(os.path.join(DATA_DIR, 'clustered_papers.csv')),
            "active_jobs": len(active_jobs),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
                "totalClusters": df['cluster'].nunique() if 'cluster' in df.columns else 0,
                "categoriesCount": df['primary_category'].nunique() if 'primary_category' in df.columns else 0,
                "clusterSizes": df['cluster'].value_counts().to_dict() if 'cluster' in df.columns else {},
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
                    "categoriesCount": df['primary_category'].nunique() if 'primary_category' in df.columns else 0,
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
        print(f"Error in get_stats: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/collect-data', methods=['POST'])
def collect_data():
    """Start data collection process"""
    try:
        # Debug logging
        print("=" * 50)
        print("📥 Received data collection request")
        print(f"Request method: {request.method}")
        print(f"Content-Type: {request.headers.get('Content-Type', 'Not set')}")
        print(f"Raw request data: {request.get_data()}")
        
        data = request.json or {}
        print(f"Parsed JSON data: {data}")
        
        categories = data.get('categories', [])
        max_results = data.get('maxResults', 1000)
        use_primary_only = data.get('usePrimaryOnly', False)
        
        print(f"Extracted categories: {categories} (type: {type(categories)}, length: {len(categories)})")
        print(f"Max results: {max_results}")
        print(f"Use primary only: {use_primary_only}")
        print("=" * 50)
        
        # Validate input
        if not categories:
            print("❌ ERROR: No categories provided")
            return jsonify({"error": "No categories provided"}), 400
        
        # Kategori sayısını sınırla (performans için)
        if len(categories) > 35:
            print(f"❌ ERROR: Too many categories: {len(categories)}")
            return jsonify({"error": "Too many categories selected. Maximum 35 categories allowed."}), 400
        
        print("✅ Validation passed, starting job...")
        
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        print(f"Starting data collection job: {job_id}")
        
        # Initialize job status
        update_job_status(job_id, "started", 0, "Initializing data collection...")
        
        # Start collection in a background thread
        def run_collection():
            try:
                update_job_status(job_id, "running", 10, "Setting up ArXiv collector...")
                
                # Progress callback function
                def progress_callback(message, percentage):
                    update_job_status(job_id, "running", percentage, message)
                
                # Initialize collector with optimized settings and progress callback
                collector = ArXivDataCollector(max_results=max_results, delay=0.2, max_workers=3, 
                                             progress_callback=progress_callback)
                
                update_job_status(job_id, "running", 20, f"Collecting papers from {len(categories)} categories...")
                
                if use_primary_only:
                    update_job_status(job_id, "running", 30, f"Using primary category filter for: {', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}")
                    # Use new method with timeout protection
                    df = collector.collect_papers_by_primary_category(categories)
                else:
                    update_job_status(job_id, "running", 30, f"Collecting from categories: {', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}")
                    # Collect papers with timeout protection
                    df = collector.collect_papers_by_category(categories)
                    
                    update_job_status(job_id, "running", 85, "Filtering by primary categories...")
                    # Filter by primary categories
                    original_count = len(df)
                    df = df[df['primary_category'].isin(categories)]
                    filtered_count = len(df)
                    
                    update_job_status(job_id, "running", 88, f"Filtered {original_count} -> {filtered_count} papers")
                
                if len(df) == 0:
                    update_job_status(job_id, "failed", 0, "No papers collected. Try fewer categories or check category names.")
                    return
                
                update_job_status(job_id, "running", 90, f"Preprocessing {len(df)} papers...")
                
                # Preprocess
                df_clean = collector.preprocess_dataframe(df)
                
                update_job_status(job_id, "running", 95, "Saving data...")
                
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
                    f"Successfully collected {len(df_clean)} papers from {df_clean['primary_category'].nunique()} categories",
                    stats)
                
            except Exception as e:
                error_msg = f"Error during collection: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                update_job_status(job_id, "failed", 0, error_msg)
        
        # Start background thread with timeout protection
        thread = threading.Thread(target=run_collection)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "jobId": job_id,
            "message": "Data collection started",
            "usePrimaryOnly": use_primary_only,
            "maxCategories": len(categories)
        })
        
    except Exception as e:
        error_msg = f"Error starting collection: {str(e)}"
        print(f"❌ CRITICAL ERROR: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route('/api/cluster', methods=['POST'])
def perform_clustering():
    """Start clustering process"""
    try:
        data = request.json or {}
        vocab_size = data.get('vocabSize', 5000)
        k_range = data.get('kRange')
        specific_k = data.get('specificK')
        
        # Validate input data file exists
        data_file = os.path.join(DATA_DIR, 'arxiv_papers.csv')
        if not os.path.exists(data_file):
            return jsonify({"error": "No data available. Please collect data first."}), 400
        
        # Validate data file has content
        try:
            test_df = pd.read_csv(data_file)
            if len(test_df) == 0:
                return jsonify({"error": "Data file is empty. Please collect data first."}), 400
            print(f"Found {len(test_df)} papers for clustering")
        except Exception as e:
            return jsonify({"error": f"Error reading data file: {str(e)}"}), 400
        
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        print(f"Starting clustering job: {job_id}")
        
        # Initialize job status
        update_job_status(job_id, "started", 0, "Initializing clustering...")
        
        # Start clustering in a background thread
        def run_clustering():
            clustering = None
            try:
                # Set up k range
                if k_range:
                    k_min, k_max = k_range
                    k_range_list = range(k_min, k_max + 1)
                    print(f"Using k range: {k_min}-{k_max}")
                else:
                    k_range_list = None
                    print(f"Using specific k: {specific_k}")
                
                update_job_status(job_id, "running", 10, "Setting up Spark session...")
                
                # Initialize clustering
                clustering = SparkTextClustering()
                
                # Data loading
                update_job_status(job_id, "running", 20, "Loading data...")
                df = clustering.load_data(data_file)
                print(f"Loaded {df.count()} papers for clustering")
                
                # Text preprocessing
                update_job_status(job_id, "running", 30, "Preprocessing text...")
                df = clustering.preprocess_text(['title', 'summary'])
                
                # Feature extraction with optimized vocab size
                update_job_status(job_id, "running", 40, "Extracting TF-IDF features...")
                df_features = clustering.create_features(vocab_size=min(vocab_size, 3000), min_df=2)
                
                # Find optimal k or use specific k
                if k_range_list:
                    update_job_status(job_id, "running", 50, f"Finding optimal k in range {k_range}...")
                    optimal_k, costs, silhouette_scores = clustering.find_optimal_k(
                        k_range=k_range_list, 
                        iterations=50
                    )
                    print(f"Found optimal k: {optimal_k}")
                else:
                    optimal_k = specific_k
                    print(f"Using specified k: {optimal_k}")
                
                # Perform clustering with optimized parameters
                update_job_status(job_id, "running", 70, f"Performing K-means clustering with k={optimal_k}...")
                df_clustered = clustering.perform_clustering(
                    k=optimal_k, 
                    max_iterations=50  # Reduced from 100
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
                result_data = {
                    "optimalK": int(optimal_k),
                    "silhouetteScore": float(getattr(clustering, 'silhouette_score', 0.0)),
                    "clusterSizes": {int(k): int(v.get('size', 0)) for k, v in cluster_analysis.items()},
                    "totalPapers": int(len(result_df) if result_df is not None else 0),
                    "clusterBalance": getattr(clustering, 'cluster_balance', {}),
                    "clusterDetails": {
                        int(k): {
                            'size': int(v.get('size', 0)),
                            'homogeneity': float(v.get('homogeneity', 0)),
                            'dominant_category': v.get('dominant_category', 'Unknown'),
                            'category_diversity': int(v.get('category_diversity', 0)),
                            'outlier_count': int(v.get('outlier_count', 0))
                        } for k, v in cluster_analysis.items()
                    }
                }
                
                update_job_status(
                    job_id, 
                    "completed", 
                    100, 
                    f"Clustering completed successfully with k={optimal_k}",
                    result_data
                )
                
                print(f"Clustering job {job_id} completed successfully")
                    
            except Exception as e:
                error_msg = f"Error during clustering: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                update_job_status(job_id, "failed", 0, error_msg)
            finally:
                # Always stop the Spark session
                if clustering:
                    try:
                        clustering.stop_spark()
                        print("Spark session stopped")
                    except Exception as e:
                        print(f"Error stopping Spark: {e}")
        
        # Start the thread
        threading.Thread(target=run_clustering, daemon=True).start()
        
        return jsonify({"jobId": job_id, "message": "Clustering started"})
        
    except Exception as e:
        error_msg = f"Error starting clustering: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route('/api/job-status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get the status of a job"""
    try:
        status = get_job_status(job_id)
        if status.get("status") == "not_found":
            return jsonify({"error": "Job not found", "jobId": job_id}), 404
        return jsonify(status)
    except Exception as e:
        print(f"Error getting job status: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/clustered-papers', methods=['GET'])
def get_clustered_papers():
    """Get the clustered papers"""
    try:
        clustered_file = os.path.join(DATA_DIR, 'clustered_papers.csv')
        if not os.path.exists(clustered_file):
            return jsonify({"error": "No clustered data available. Please run clustering first."}), 404
        
        # Add pagination support
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 100, type=int)
        search = request.args.get('search', '', type=str)
        cluster_filter = request.args.get('cluster', None, type=int)
        
        df = pd.read_csv(clustered_file)
        
        # Apply filters
        if search:
            mask = df['title'].str.contains(search, case=False, na=False) | \
                   df['summary'].str.contains(search, case=False, na=False)
            df = df[mask]
        
        if cluster_filter is not None:
            df = df[df['cluster'] == cluster_filter]
        
        # Calculate pagination
        total = len(df)
        start = (page - 1) * per_page
        end = start + per_page
        
        # Get page data
        page_df = df.iloc[start:end]
        papers = page_df.to_dict(orient='records')
        
        return jsonify({
            "papers": papers,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            },
            "filters": {
                "search": search,
                "cluster": cluster_filter
            }
        })
    except Exception as e:
        print(f"Error in get_clustered_papers: {e}")
        traceback.print_exc()
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
            "wordcloudsUrl": "cluster_wordclouds.png",
            "optimalKUrl": "optimal_k_analysis.png"
        }
        
        for url_key, filename in files_to_check.items():
            file_path = os.path.join(VISUALIZATIONS_DIR, filename)
            if os.path.exists(file_path):
                visualizations[url_key] = f"/visualizations/{filename}"
                # Add file info
                stat = os.stat(file_path)
                visualizations[f"{url_key}_info"] = {
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                visualizations[url_key] = None
        
        # Also return file existence status
        visualizations["filesExist"] = {
            key: visualizations[key] is not None for key in files_to_check.keys()
        }
        
        # Count total files
        visualizations["totalFiles"] = sum(1 for v in visualizations["filesExist"].values() if v)
        
        return jsonify(visualizations)
    except Exception as e:
        print(f"Error in get_visualizations: {e}")
        traceback.print_exc()
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
    print("🚀 Starting Academic Paper Clustering API...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📊 Visualizations directory: {VISUALIZATIONS_DIR}")
    print(f"💼 Jobs directory: {JOBS_DIR}")
    print(f"🌐 Server will be available at: http://localhost:5000")
    print(f"🔧 Debug mode: True")
    app.run(debug=True, host='0.0.0.0', port=5000) 
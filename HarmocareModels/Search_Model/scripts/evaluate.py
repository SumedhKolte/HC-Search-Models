#specialized for the medical search engine project
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import os
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.database import Database
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger
from config.settings import DB_CONFIG, MODEL_CONFIG

# Setup logging
logger = setup_logger(__name__)

def setup_ssl() -> str:
    """Setup SSL certificate for database connection with timestamp tracking"""
    ssl_dir = project_root / 'ssl'
    ssl_dir.mkdir(exist_ok=True)
    cert_path = ssl_dir / 'rds-ca-bundle.pem'
    timestamp_path = ssl_dir / 'cert_timestamp.txt'
    
    current_time = datetime.now(timezone.utc)
    
    needs_update = True
    if cert_path.exists() and timestamp_path.exists():
        with open(timestamp_path, 'r') as f:
            last_update = datetime.fromisoformat(f.read().strip())
            if (current_time - last_update).days < 30:
                needs_update = False
    
    if needs_update:
        logger.info(f"Downloading SSL certificate at {current_time.isoformat()}")
        response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
        response.raise_for_status()
        
        with open(cert_path, 'wb') as f:
            f.write(response.content)
        
        with open(timestamp_path, 'w') as f:
            f.write(current_time.isoformat())
    
    return str(cert_path)

def evaluate_search_metrics(db: Database) -> Dict:
    """Evaluate search performance metrics"""
    metrics = {}
    
    # Query success rates
    query = """
    SELECT 
        search_type,
        COUNT(*) as total_searches,
        AVG(CASE WHEN results_count > 0 THEN 1 ELSE 0 END) as success_rate,
        AVG(execution_time) as avg_execution_time,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY execution_time) as p95_latency
    FROM search_metrics
    GROUP BY search_type
    """
    search_metrics = pd.read_sql(query, db.engine)
    metrics['search_performance'] = search_metrics.to_dict('records')
    
    # Query expansion effectiveness
    query = """
    SELECT 
        qe.terms_added,
        AVG(sm.results_count) as avg_results,
        AVG(sm.execution_time) as avg_execution_time
    FROM query_expansion_logs qe
    JOIN search_metrics sm ON qe.original_query = sm.query
    GROUP BY qe.terms_added
    ORDER BY qe.terms_added
    """
    expansion_metrics = pd.read_sql(query, db.engine)
    metrics['query_expansion'] = expansion_metrics.to_dict('records')
    
    return metrics

def evaluate_model_metrics(db: Database, model: SentenceTransformer) -> Dict:
    """Evaluate model performance metrics"""
    metrics = {}
    
    # Model training metrics
    query = """
    SELECT 
        entity_type,
        AVG(executiontime) as avg_training_time,
        SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
    FROM model_training_metrics
    GROUP BY entity_type
    """
    model_metrics = pd.read_sql(query, db.engine)
    metrics['training'] = model_metrics.to_dict('records')
    
    return metrics

def generate_evaluation_plots(metrics: Dict):
    """Generate evaluation plots"""
    output_dir = project_root / 'monitoring' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # Search performance plot
    plt.figure(figsize=(10, 6))
    search_df = pd.DataFrame(metrics['search_performance'])
    sns.barplot(data=search_df, x='search_type', y='success_rate')
    plt.title('Search Success Rate by Type')
    plt.tight_layout()
    plt.savefig(output_dir / f'search_performance_{timestamp}.png')
    
    # Query expansion plot
    plt.figure(figsize=(10, 6))
    expansion_df = pd.DataFrame(metrics['query_expansion'])
    sns.lineplot(data=expansion_df, x='terms_added', y='avg_results')
    plt.title('Query Expansion Effectiveness')
    plt.tight_layout()
    plt.savefig(output_dir / f'query_expansion_{timestamp}.png')

def save_evaluation_results(metrics: Dict):
    """Save evaluation results with timestamp"""
    output_dir = project_root / 'monitoring' / 'metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    metrics['timestamp'] = timestamp
    
    output_file = output_dir / f'evaluation_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_file}")

def main():
    """Main evaluation function"""
    logger.info("Starting evaluation process")
    
    # Setup SSL
    ssl_cert_path = setup_ssl()
    
    try:
        # Initialize database connection
        db = Database(ssl_cert_path=ssl_cert_path)
        
        # Load model
        model = SentenceTransformer(MODEL_CONFIG['base_model'])
        
        # Collect metrics
        search_metrics = evaluate_search_metrics(db)
        model_metrics = evaluate_model_metrics(db, model)
        
        # Combine metrics
        all_metrics = {
            'search': search_metrics,
            'model': model_metrics
        }
        
        # Generate plots
        generate_evaluation_plots(all_metrics)
        
        # Save results
        save_evaluation_results(all_metrics)
        
        # Log success
        db.engine.execute(
            "INSERT INTO system_updates (status, error_message) VALUES (true, 'Evaluation completed successfully')"
        )
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        logger.error(error_msg)
        
        if 'db' in locals():
            db.engine.execute(
                "INSERT INTO system_updates (status, error_message) VALUES (false, %s)",
                [error_msg]
            )

if __name__ == "__main__":
    main()
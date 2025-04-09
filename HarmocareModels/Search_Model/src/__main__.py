# Accurate AI-based search

# Dynamic data ingestion

# Real-time updates to vector DB or FAISS



import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.tasks.update_embeddings import run_embedding_updates
import asyncio

if __name__ == "__main__":
    asyncio.run(run_embedding_updates())
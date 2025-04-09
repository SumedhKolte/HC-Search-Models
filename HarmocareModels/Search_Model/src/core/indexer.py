import logging
from pathlib import Path
from typing import Dict, List, Optional
import faiss
import numpy as np
import json
from datetime import datetime, timezone

from src.utils.logger import setup_logger
from config.settings import MODEL_CONFIG

logger = setup_logger(__name__)

class IndexManager:
    """FAISS index management"""
    
    def __init__(self):
        """Initialize index manager"""
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.index_dir = self.project_root / 'ml' / 'indexes'
        self.indexes: Dict[str, faiss.Index] = {}
        self.id_mappings: Dict[str, Dict[int, str]] = {}
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def get_index(self, entity_type: str) -> faiss.Index:
        """Get FAISS index for entity type"""
        if entity_type not in self.indexes:
            self._load_index(entity_type)
        return self.indexes[entity_type]

    def get_id_mapping(self, entity_type: str) -> Dict[int, str]:
        """Get ID mapping for entity type"""
        if entity_type not in self.id_mappings:
            self._load_index(entity_type)
        return self.id_mappings[entity_type]

    def _load_index(self, entity_type: str) -> None:
        """Load FAISS index and ID mapping from disk"""
        entity_dir = self.index_dir / entity_type
        
        if not entity_dir.exists():
            raise ValueError(f"No index directory found for {entity_type}")

        # Get latest index and mapping files
        index_path = entity_dir / 'latest.faiss'
        mapping_path = entity_dir / 'latest_mapping.json'

        if not index_path.exists() or not mapping_path.exists():
            raise ValueError(f"Missing index or mapping file for {entity_type}")

        try:
            # Load index
            self.indexes[entity_type] = faiss.read_index(str(index_path))

            # Load ID mapping
            with open(mapping_path, 'r') as f:
                self.id_mappings[entity_type] = {
                    int(k): v for k, v in json.load(f)['ids'].items()
                }

        except Exception as e:
            logger.error(f"Failed to load index for {entity_type}: {str(e)}")
            raise

    def update_index(self,
                    entity_type: str,
                    embeddings: np.ndarray,
                    ids: List[str],
                    index_type: str = 'IVFFlat') -> None:
        """Create/update FAISS index for entity type"""
        try:
            timestamp = datetime.now(timezone.utc)
            dimension = embeddings.shape[1]

            # Create appropriate index
            if index_type == 'IVFFlat':
                nlist = min(4096, max(int(np.sqrt(len(embeddings))), 1))
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                
                # Train index
                if len(embeddings) > nlist:
                    index.train(embeddings)
            else:
                index = faiss.IndexFlatL2(dimension)

            # Add vectors
            index.add(embeddings)

            # Save index and mapping
            entity_dir = self.index_dir / entity_type
            entity_dir.mkdir(parents=True, exist_ok=True)

            # Save with timestamp
            index_path = entity_dir / f'index_{timestamp:%Y%m%d_%H%M%S}.faiss'
            mapping_path = entity_dir / f'id_mapping_{timestamp:%Y%m%d_%H%M%S}.json'

            # Save index
            faiss.write_index(index, str(index_path))

            # Save ID mapping
            with open(mapping_path, 'w') as f:
                json.dump({'ids': {i: id_ for i, id_ in enumerate(ids)}}, f)

            # Update symlinks
            latest_index = entity_dir / 'latest.faiss'
            latest_mapping = entity_dir / 'latest_mapping.json'

            if latest_index.exists():
                latest_index.unlink()
            if latest_mapping.exists():
                latest_mapping.unlink()

            latest_index.symlink_to(index_path)
            latest_mapping.symlink_to(mapping_path)

            # Update in-memory
            self.indexes[entity_type] = index
            self.id_mappings[entity_type] = {i: id_ for i, id_ in enumerate(ids)}

            logger.info(f"Updated index for {entity_type} at {timestamp.isoformat()}")

        except Exception as e:
            logger.error(f"Failed to update index for {entity_type}: {str(e)}")
            raise
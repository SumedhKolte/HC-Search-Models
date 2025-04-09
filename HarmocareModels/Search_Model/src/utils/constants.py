"""Constants used throughout the medical search application"""

EMBEDDING_MODELS = {
    'default': 'sentence-transformers/all-MiniLM-L6-v2',  # Changed default model
    'bge_small': 'BAAI/bge-small-en-v1.5',
    'bge_base': 'BAAI/bge-base-en-v1.5',
    'bge_large': 'BAAI/bge-large-en-v1.5'
}

ENTITY_CONFIGS = {
    'doctors': {
        'table_name': 'doctors',
        'id_column': 'did',
        'text_column': 'name',  # Primary text column for searching
        'embedding_column': 'embedding',
        'text_columns': ['name', 'specialization', 'experience'],  # All text columns for embedding
        'vector_dim': 384
    },
    'hospitals': {
        'table_name': 'hospitals',
        'id_column': 'hid',
        'text_column': 'name',
        'embedding_column': 'embedding',
        'text_columns': ['name', 'location', 'hospital_type'],
        'vector_dim': 384
    },
    'clinics': {
        'table_name': 'clinics',
        'id_column': 'cid',
        'text_column': 'name',
        'embedding_column': 'embedding',
        'text_columns': ['name', 'location'],
        'vector_dim': 384
    },
    'diseases': {
        'table_name': 'diseases',
        'id_column': 'disease_id',
        'text_column': 'name',
        'embedding_column': 'embedding',
        'text_columns': ['name', 'description'],
        'vector_dim': 384
    },
    'symptoms': {
        'table_name': 'symptoms',
        'id_column': 'symptom_id',
        'text_column': 'name',
        'embedding_column': 'embedding',
        'text_columns': ['name', 'description'],
        'vector_dim': 384
    }
}

# Search configuration
SEARCH_CONFIG = {
    'min_score': 0.5,  # Minimum similarity score threshold
    'max_results': 50,  # Maximum number of results to return
    'default_k': 10    # Default number of results to return
}

# Vector dimensions for different models
VECTOR_DIMS = {
    'default': 384,  # all-MiniLM-L6-v2 dimension
    'bge_small': 384,
    'bge_base': 768,
    'bge_large': 1024
}
"""
NLP package for medical search system
Provides text processing, query expansion, and analysis functionality
"""

from .processor import TextProcessor
from .expander import QueryExpander 
from .analyzer import QueryAnalyzer

__all__ = [
    'TextProcessor',
    'QueryExpander',
    'QueryAnalyzer'
]

# Version info
__version__ = '1.0.0'

# The `QueryProcessor` class provides:
# 1. Query cleaning and normalization
# 2. Embedding generation
# 3. Query expansion (placeholder)
# 4. Query logging (placeholder)

# Now run the server again:
# ```powershell
# cd c:\search\b2\medical_search
# uvicorn src.api.main:app --reload --port 8000
# ```

# The import error should be resolved and the server should start correctly.
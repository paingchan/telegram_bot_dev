import os
from dotenv import load_dotenv
import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Force reload environment variables
load_dotenv(override=True)

# Clear any cached instances
_model = None
_qdrant_client = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model...")
        _model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return _model

def get_qdrant_client():
    """Get a fresh Qdrant client instance with current environment variables"""
    global _qdrant_client
    
    # Force clear any existing client
    _qdrant_client = None
    
    # Get fresh environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variables are not set")
        
    logger.info("Initializing Qdrant client...")
    _qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=10
    )
    return _qdrant_client 
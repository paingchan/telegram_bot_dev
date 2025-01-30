import os
import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Lazy loading for SentenceTransformer
@lru_cache(maxsize=1)
def get_model():
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

# Lazy loading for Qdrant client
@lru_cache(maxsize=1)
def get_qdrant_client():
    logger.info("Initializing Qdrant client...")
    return QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    ) 
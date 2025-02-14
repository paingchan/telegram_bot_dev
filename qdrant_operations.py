import os
from dotenv import load_dotenv
import json
from qdrant_client.http.models import Distance, VectorParams
import logging
import httpx
from utils import get_qdrant_client, get_model

# Force reload environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get fresh environment variables
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variables are not set")

logger.info(f"Using Qdrant URL: {QDRANT_URL}")  # Log the URL being used

# Create Qdrant collections if they don't exist
def check_collection_exists(collection_name):
    """Check if a collection exists in Qdrant"""
    try:
        qdrant_client = get_qdrant_client()
        collections = qdrant_client.get_collections()
        return collection_name in [col.name for col in collections.collections]
    except Exception as e:
        logger.error(f"Error checking collection {collection_name}: {e}")
        return False

def create_collection_if_not_exists(collection_name):
    """Create collection only if it doesn't exist"""
    try:
        if not check_collection_exists(collection_name):
            logger.info(f"Creating new collection: {collection_name}")
            qdrant_client = get_qdrant_client()
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info(f"Successfully created collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")

# Generate embeddings and upsert data into Qdrant
def upsert_data(collection_name, data, id_field="id", question_field="question", answer_field="answer"):
    model = get_model()
    qdrant_client = get_qdrant_client()
    
    questions = [item[question_field] for item in data]
    question_embeddings = model.encode(questions)

    points = [
        {
            "id": item[id_field],
            "vector": embedding.tolist(),
            "payload": {
                "question": item[question_field],
                "answer": item[answer_field],
            },
        }
        for item, embedding in zip(data, question_embeddings)
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Data upserted into Qdrant collection: {collection_name}")

# Function to search Qdrant collection
def search_collection(collection_name, query_embedding, limit=1):
    qdrant_client = get_qdrant_client()  # Lazy load the client
    logger.info(f"Searching in collection: {collection_name} with limit: {limit}")
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit,
    )
    if results:
        logger.info(f"Search results: {results}")
    else:
        logger.warning("No results found.")
    return results

async def search_collection_async(collection_name: str, query_vector: list, limit: int = 1):
    """Asynchronous version of search_collection"""
    try:
        # Force get fresh URL from environment
        qdrant_url = os.getenv("QDRANT_URL", "").rstrip('/')
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("Qdrant configuration missing")
            
        logger.info(f"Making request to Qdrant URL: {qdrant_url}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{qdrant_url}/collections/{collection_name}/points/search",
                headers={
                    "api-key": qdrant_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "vector": query_vector,
                    "limit": limit,
                    "with_payload": True,
                    "with_vectors": False
                },
                timeout=httpx.Timeout(10.0)
            )
            response.raise_for_status()
            results = response.json().get("result", [])
            return results
    except Exception as e:
        logger.error(f"Error searching collection {collection_name}: {e}")
        logger.error(f"Using URL: {qdrant_url}")
        return []
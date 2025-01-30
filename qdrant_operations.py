import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables for API keys
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Create Qdrant collections if they don't exist
def create_collection(collection_name):
    collections = qdrant_client.get_collections()
    if collection_name not in [col.name for col in collections.collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {collection_name}")

# Generate embeddings and upsert data into Qdrant
def upsert_data(collection_name, data, id_field="id", question_field="question", answer_field="answer"):
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

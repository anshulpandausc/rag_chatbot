from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv
import os
import uuid

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

COLLECTION_NAME = "arxiv_data"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output size

_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = QdrantClient(
    url="https://a4fd2c51-972b-4f22-87a8-fd40f5187b84.us-east-1-1.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wmXhkKXYHkj80irmS9OafBjzTiTGRaI5bdWAQ26rydM",
)


def ensure_collection():
    """Creates the Qdrant collection if it doesn't already exist."""
    existing = [c.name for c in _client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            optimizers_config={"indexing_threshold": 0}
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")


def embed_and_insert(summary: str, chunk_text: str, metadata: dict, chunk_index: int):
    """Embeds a summary and inserts it into Qdrant with full metadata payload."""
    vector = _model.encode(summary).tolist()

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "paper_id": metadata.get("paper_id"),
            "title": metadata.get("title"),
            "authors": metadata.get("authors"),
            "categories": metadata.get("categories"),
            "update_date": metadata.get("update_date"),
            "chunk_type": metadata.get("chunk_type"),
            "chunk_text": chunk_text,
            "summary": summary,
            "chunk_index": chunk_index,
        }
    )

    _client.upsert(collection_name=COLLECTION_NAME, points=[point])
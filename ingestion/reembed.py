from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
import os
import time
import uuid

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

COLLECTION_NAME = "arxiv_data"
NEW_VECTOR_SIZE = 768  # BAAI/bge-base-en-v1.5 output size
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64

qdrant = QdrantClient(
    url="https://a4fd2c51-972b-4f22-87a8-fd40f5187b84.us-east-1-1.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wmXhkKXYHkj80irmS9OafBjzTiTGRaI5bdWAQ26rydM",
    timeout=120
)

model = SentenceTransformer(EMBEDDING_MODEL)


def fetch_all_chunks() -> list:
    """Fetches all chunks from Qdrant."""
    chunks = []
    limit = 100
    offset = None
    while True:
        response = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        points, offset = response
        chunks.extend(points)
        if offset is None:
            break
    return chunks


if __name__ == "__main__":
    start = time.time()

    # Step 1: Fetch all existing chunks
    print("Fetching all chunks from Qdrant...")
    chunks = fetch_all_chunks()
    print(f"Fetched {len(chunks)} chunks.\n")

    # Step 2: Recreate collection with new vector size
    print(f"Recreating collection with vector size {NEW_VECTOR_SIZE}...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=NEW_VECTOR_SIZE, distance=Distance.COSINE),
        optimizers_config={"indexing_threshold": 0}
    )
    print("Collection recreated.\n")

    # Step 3: Re-embed summaries and upsert in batches
    print(f"Re-embedding with {EMBEDDING_MODEL}...")
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]

        summaries = [
            f"Represent this sentence for searching relevant scientific papers: {p.payload.get('summary', '')}"
            for p in batch
        ]

        vectors = model.encode(summaries, normalize_embeddings=True).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload=batch[i].payload
            )
            for i in range(len(batch))
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  → Upserted {batch_start + len(batch)}/{len(chunks)} chunks")

    elapsed = time.time() - start
    print(f"\nRe-embedding complete in {elapsed:.2f} seconds.")
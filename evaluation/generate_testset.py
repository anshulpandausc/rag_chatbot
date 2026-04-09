from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from collections import defaultdict
import os
import random
import json

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

openai_api_key = os.getenv("KEY")

COLLECTION_NAME = "arxiv_data"
MAX_CHUNKS_PER_PAPER = 3

# ---------------- CLIENTS ----------------
qdrant = QdrantClient(
    url="https://a4fd2c51-972b-4f22-87a8-fd40f5187b84.us-east-1-1.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wmXhkKXYHkj80irmS9OafBjzTiTGRaI5bdWAQ26rydM",
    timeout=120
)
openai_client = OpenAI(api_key=openai_api_key)


def fetch_all_chunks(collection_name: str) -> list[dict]:
    """Fetches all chunks from Qdrant."""
    chunks = []
    limit = 100
    offset = None

    while True:
        response = qdrant.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        points, offset = response
        for point in points:
            chunks.append(point.payload)
        if offset is None:
            break

    return chunks


def sample_chunks_from_paper(paper_chunks: list[dict], n: int) -> list[dict]:
    """
    Samples n chunks evenly spread across a paper's chunks by position.
    Ensures coverage of early, middle, and late sections.
    """
    if len(paper_chunks) <= n:
        return paper_chunks

    # Sort by chunk_index to preserve document order
    sorted_chunks = sorted(paper_chunks, key=lambda c: c.get("chunk_index", 0))

    # Pick evenly spaced indices across the document
    indices = [int(i * (len(sorted_chunks) - 1) / (n - 1)) for i in range(n)]
    return [sorted_chunks[i] for i in indices]


def generate_question_and_answer(chunk_text: str, summary: str, title: str) -> dict | None:
    """Generates a question and ground truth answer from a chunk."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an academic assistant creating evaluation data for a RAG system. "
                    "Given a research paper chunk and its summary, generate one specific question "
                    "that can be answered using the chunk, and provide the ground truth answer. "
                    "The question should reflect a realistic user query — specific enough to require "
                    "the chunk to answer, but natural enough that a researcher might actually ask it. "
                    "Respond ONLY with a JSON object with two fields: 'question' and 'ground_truth'. "
                    "Do not include any preamble or markdown formatting."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Paper title: {title}\n\n"
                    f"Summary: {summary}\n\n"
                    f"Full chunk:\n{chunk_text}"
                )
            }
        ],
        max_tokens=300,
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        if "question" in parsed and "ground_truth" in parsed:
            return parsed
    except json.JSONDecodeError:
        print(f"  ⚠ Failed to parse response: {raw[:100]}")
    return None


if __name__ == "__main__":
    print("Fetching chunks from Qdrant...")
    chunks = fetch_all_chunks(COLLECTION_NAME)
    print(f"Fetched {len(chunks)} chunks.\n")

    # Filter to body chunks only
    body_chunks = [
        c for c in chunks
        if c.get("chunk_text", "").strip()
        and c.get("summary", "").strip()
        and c.get("chunk_type") == "body"
    ]
    print(f"Filtered to {len(body_chunks)} body chunks.\n")

    # Group by paper
    paper_buckets = defaultdict(list)
    for chunk in body_chunks:
        paper_buckets[chunk["paper_id"]].append(chunk)

    # Sample evenly spaced chunks from each paper
    sampled_chunks = []
    for paper_id, paper_chunks in paper_buckets.items():
        sampled_chunks.extend(sample_chunks_from_paper(paper_chunks, MAX_CHUNKS_PER_PAPER))

    print(f"Sampled {len(sampled_chunks)} chunks across {len(paper_buckets)} papers.\n")

    # Generate questions and ground truth answers
    testset = []
    for i, chunk in enumerate(sampled_chunks, start=1):
        print(f"[{i}/{len(sampled_chunks)}] Generating question for: {chunk.get('title', '')[:60]}")
        result = generate_question_and_answer(
            chunk_text=chunk.get("chunk_text", ""),
            summary=chunk.get("summary", ""),
            title=chunk.get("title", "")
        )
        if result:
            testset.append({
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "paper_id": chunk.get("paper_id", ""),
                "title": chunk.get("title", ""),
                "chunk_index": chunk.get("chunk_index"),
                "source_chunk": chunk.get("chunk_text", ""),
            })

    print(f"\nGenerated {len(testset)} question-answer pairs.\n")

    # Save testset
    output_path = Path(__file__).resolve().parent / "data" / "testset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2)

    print(f"Testset saved to {output_path}")
    print(f"\nSample:")
    print(f"Q: {testset[0]['question']}")
    print(f"A: {testset[0]['ground_truth']}")
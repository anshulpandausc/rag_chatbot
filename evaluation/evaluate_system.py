from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
import os
import json
from pathlib import Path

# ---------------- ENV ----------------
load_dotenv()
openai_api_key = os.getenv("KEY")

# ---------------- CONFIG ----------------
COLLECTION_NAME = "arxiv_data"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5
SCORE_THRESHOLD = 0.5

# ---------------- MODELS ----------------
embed_model = SentenceTransformer(EMBEDDING_MODEL)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key
)

# ---------------- QDRANT ----------------
qdrant = QdrantClient(
    url="https://a4fd2c51-972b-4f22-87a8-fd40f5187b84.us-east-1-1.aws.cloud.qdrant.io:6333",
    api_key="YOUR_QDRANT_API_KEY",
    timeout=120
)

# ---------------- RETRIEVAL ----------------
def retrieve(query: str, keywords: list[str] = None) -> list[str]:
    prefixed_query = f"Represent this sentence for searching relevant scientific papers: {query}"
    embedding = embed_model.encode(prefixed_query, normalize_embeddings=True).tolist()

    if keywords is None:
        keywords = [w.lower() for w in query.split() if len(w) > 4]

    search_filter = None
    if keywords:
        conditions = [
            FieldCondition(key="chunk_text", match=MatchText(text=kw))
            for kw in keywords
        ]
        search_filter = Filter(should=conditions)

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        query_filter=search_filter,
        limit=TOP_K,
    )
    hits = [h for h in response.points if h.score >= SCORE_THRESHOLD]

    # fallback if nothing passes threshold
    if not hits:
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=TOP_K,
        )
        hits = [h for h in response.points if h.score >= SCORE_THRESHOLD]

    return [h.payload.get("chunk_text", "") for h in hits]

# ---------------- GENERATION ----------------
def generate_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "No relevant information found."

    context_text = "\n\n".join(
        f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)
    )

    prompt = f"""You are an academic assistant. Use the following research paper chunks to answer the question.
Be concise and factual.

Question: {query}

Chunks:
{context_text}
"""
    response = llm.invoke(prompt)
    return response.content

# ---------------- HIT RATE ----------------
def compute_hit_rate(contexts: list[str], ground_truth: str) -> int:
    ground_truth_lower = ground_truth.lower()
    for ctx in contexts:
        if ground_truth_lower[:50] in ctx.lower():
            return 1
    return 0

# ---------------- MAIN ----------------
if __name__ == "__main__":
    testset_path = Path(__file__).resolve().parent / "data" / "testset.json"

    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    print(f"Evaluating full testset with {len(testset)} questions.\n")

    samples = []
    hit_scores = []

    for i, item in enumerate(testset, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        if isinstance(ground_truth, list):
            ground_truth = " ".join(ground_truth)

        print(f"[{i}/{len(testset)}] {question[:80]}")

        contexts = retrieve(question)
        answer = generate_answer(question, contexts)

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
            reference=ground_truth,
        ))

        hit_scores.append(compute_hit_rate(contexts, ground_truth))

    # ---------------- RAGAS ----------------
    dataset = EvaluationDataset(samples=samples)

    print("\nRunning Faithfulness evaluation...\n")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=llm,
    )

    df = results.to_pandas()

    # ---------------- FINAL METRICS ----------------
    avg_faithfulness = df["faithfulness"].mean()
    hit_rate = sum(hit_scores) / len(hit_scores)

    print("\n================ FINAL SCORES ================\n")
    print(f"Faithfulness: {avg_faithfulness:.4f}")
    print(f"Hit Rate: {hit_rate:.4f}")

    # Save results
    output_path = Path(__file__).resolve().parent / "data" / "evaluation_results.json"
    df["hit_rate"] = hit_scores
    df.to_json(output_path, orient="records", indent=2)

    print(f"\nDetailed results saved to {output_path}")

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer
import os

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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# ---------------- QDRANT ----------------
client = QdrantClient(
    url="https://a4fd2c51-972b-4f22-87a8-fd40f5187b84.us-east-1-1.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wmXhkKXYHkj80irmS9OafBjzTiTGRaI5bdWAQ26rydM",
    timeout=120
)

# ---------------- QUERY EMBEDDING ----------------
def embed_query(query: str):
    prefixed_query = f"Represent this sentence for searching relevant scientific papers: {query}"
    embedding = embed_model.encode(prefixed_query, normalize_embeddings=True)
    return embedding.tolist()

# ---------------- RETRIEVAL ----------------
def retrieve_papers(query: str, keywords: list[str] = None):
    query_vector = embed_query(query)

    if keywords is None:
        keywords = [w.lower() for w in query.split() if len(w) > 4]

    search_filter = None
    if keywords:
        conditions = [
            FieldCondition(key="chunk_text", match=MatchText(text=kw))
            for kw in keywords
        ]
        search_filter = Filter(should=conditions)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=TOP_K,
    )
    hits = [h for h in response.points if h.score >= SCORE_THRESHOLD]

    # Fall back to unfiltered search if no hits
    if not hits:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
        )
        hits = [h for h in response.points if h.score >= SCORE_THRESHOLD]

    return hits

# ---------------- GPT SUMMARIZATION ----------------
def summarize_abstracts(hits, user_question):
    if not hits:
        return "No high-confidence matches found."

    context_text = ""
    for i, hit in enumerate(hits, start=1):
        payload = hit.payload
        link = f"https://arxiv.org/pdf/{payload['paper_id']}"
        context_text += (
            f"[{i}] Title: {payload['title']}\n"
            f"Paper ID: {payload['paper_id']} ({link})\n"
            f"Chunk: {payload['chunk_text']}\n\n"
        )

    prompt = f"""
You are an academic assistant. Use the following research paper chunks to answer the user's question.

Structure your response exactly as follows:

ANSWER:
A thorough, self-contained answer to the user's question based on the retrieved chunks.

RELEVANT PAPERS:
For each relevant paper, list its title, a link, and 1-2 sentences on how it directly relates to the question.

User Question: {user_question}

Chunks:
{context_text}
"""

    response = llm.invoke(prompt)
    return response.content

# ---------------- PRINT RESULTS ----------------
def print_results(hits):
    print("\n================ RETRIEVED CHUNKS ================\n")
    if not hits:
        print("No high-confidence matches found.\n")
        return
    for i, hit in enumerate(hits):
        payload = hit.payload
        link = f"https://arxiv.org/pdf/{payload['paper_id']}"
        print(f"RESULT {i+1}")
        print(f"Score: {hit.score:.4f}")
        print(f"Title: {payload['title']}")
        print(f"Paper ID: {payload['paper_id']} ({link})")
        print(f"Categories: {payload.get('categories')}")
        print(f"Chunk Type: {payload.get('chunk_type')}")
        print(f"Chunk Index: {payload.get('chunk_index')}")
        print(f"Summary: {payload.get('summary')}")
        print("\nChunk Text:")
        print(payload['chunk_text'][:1000])
        print("\n------------------------------------------------\n")

# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    user_query = input("\nEnter your research question: ")

    hits = retrieve_papers(user_query)
    print_results(hits)

    print("\n================ SUMMARY ANSWER =================\n")
    answer = summarize_abstracts(hits, user_query)
    print(answer)

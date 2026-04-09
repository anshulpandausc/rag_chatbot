import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from ingestion.load_papers import load_papers
from ingestion.download_pdf import download_pdf
from ingestion.chunk_pdf import chunk_pdf
from ingestion.summarize import summarize_chunk
from ingestion.embed_and_insert import ensure_collection, embed_and_insert
import time


if __name__ == "__main__":
    start_time = time.time()

    json_path = Path(__file__).resolve().parent.parent / "data" / "papers.json"
    papers = load_papers(str(json_path))
    total_papers = len(papers)
    print(f"Loaded {total_papers} papers.\n")

    ensure_collection()

    for paper_num, paper in enumerate(papers, start=1):
        print(f"[{paper_num}/{total_papers}] Processing: {paper['title']}")

        metadata = {
            "paper_id": paper["id"],
            "title": paper["title"],
            "authors": paper.get("authors"),
            "categories": paper.get("categories"),
            "update_date": paper.get("update_date"),
        }

        # Step 1: Insert abstract as its own chunk (no GPT call needed)
        print(f"  → Inserting abstract chunk")
        embed_and_insert(
            summary=paper["abstract"],
            chunk_text=paper["abstract"],
            metadata={**metadata, "chunk_type": "abstract"},
            chunk_index=0,
        )

        # Step 2: Download PDF
        try:
            pdf_path = download_pdf(paper["id"])
        except RuntimeError as e:
            print(f"  → Skipping body chunks: {e}\n")
            continue

        # Step 3: Chunk the PDF body, skipping the first page (abstract page)
        chunks = chunk_pdf(pdf_path, skip_pages=1)
        print(f"  → {len(chunks)} body chunks")

        # Step 4: Summarize + embed + insert each body chunk
        for i, chunk_text in enumerate(chunks, start=1):
            summary = summarize_chunk(
                chunk_text,
                title=paper["title"],
                authors=paper.get("authors", "")
            )
            embed_and_insert(
                summary=summary,
                chunk_text=chunk_text,
                metadata={**metadata, "chunk_type": "body"},
                chunk_index=i,
            )

        print(f"  → Done\n")

    elapsed = time.time() - start_time
    print(f"Pipeline complete in {elapsed:.2f} seconds.")
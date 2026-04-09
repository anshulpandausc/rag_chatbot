from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(api_key=os.getenv("KEY"))

def summarize_chunk(chunk_text: str, title: str = "", authors: str = "") -> str:
    """Summarizes a chunk into 2 sentences using gpt-4o-mini."""
    context = f"This chunk is from a research paper titled '{title}' by {authors}." if title else ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Summarize the following text "
                    "from a research paper into exactly 2 clear, informative sentences. "
                    "Focus on the key idea or finding. "
                    "Do not refer to 'the chunk' or 'this section' — write as if summarizing a standalone idea."
                ),
            },
            {"role": "user", "content": f"{context}\n\n{chunk_text}".strip()},
        ],
        max_tokens=100,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
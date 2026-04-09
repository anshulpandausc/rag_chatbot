import requests
from pathlib import Path


def download_pdf(paper_id: str, download_dir: str = "data/pdfs") -> str:
    """Downloads a PDF from arxiv using the paper id. Returns the local file path."""
    pdf_dir = Path(download_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{paper_id.replace('/', '_')}.pdf"

    # Skip download if already exists
    if pdf_path.exists():
        print(f"  → PDF already downloaded: {pdf_path}")
        return str(pdf_path)

    url = f"https://arxiv.org/pdf/{paper_id}"
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download PDF for {paper_id}: HTTP {response.status_code}")

    with open(pdf_path, "wb") as f:
        f.write(response.content)

    print(f"  → Downloaded PDF: {pdf_path}")
    return str(pdf_path)
import json


def load_papers(json_path: str) -> list[dict]:
    """Reads paper metadata from a JSON lines file."""
    papers = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers
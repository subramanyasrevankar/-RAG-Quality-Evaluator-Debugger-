import re


def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            while end < len(text) and text[end] != " ":
                end += 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def get_chunk_stats(chunks: list) -> dict:
    if not chunks:
        return {}
    lengths = [len(c) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chunk_length": round(sum(lengths) / len(lengths), 1),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
    }
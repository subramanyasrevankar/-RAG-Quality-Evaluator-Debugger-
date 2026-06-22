def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits a document into overlapping chunks.

    Why overlapping?
    - Sentences at chunk boundaries don't get cut off
    - Retrieval finds relevant content even if it spans two chunks

    Args:
        text       : raw document text
        chunk_size : number of characters per chunk (default 500)
        overlap    : characters repeated between chunks (default 50)

    Returns:
        list of text chunks
    """

    text = clean_text(text)

    if len(text) == 0:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Don't cut in the middle of a word — move end to next space
        if end < len(text):
            while end < len(text) and text[end] != " ":
                end += 1

        chunk = text[start:end].strip()

        if chunk:  # skip empty chunks
            chunks.append(chunk)

        # Move start forward, but overlap back by 'overlap' chars
        start = end - overlap

    return chunks


def clean_text(text: str) -> str:
    """
    Cleans raw document text before chunking.

    - Removes extra whitespace and blank lines
    - Normalizes newlines to spaces
    - Strips leading/trailing whitespace
    """
    import re

    # Replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Strip
    text = text.strip()

    return text


def get_chunk_stats(chunks: list[str]) -> dict:
    """
    Returns stats about the chunks — useful for the dashboard later.
    Interviewers love that you thought about observability.
    """
    if not chunks:
        return {}

    lengths = [len(c) for c in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_chunk_length": round(sum(lengths) / len(lengths), 1),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
    }
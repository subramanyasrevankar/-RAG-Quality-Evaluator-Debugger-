from sentence_transformers import SentenceTransformer, util
import torch

# Same model as retriever — consistency matters
# One model loaded once, reused across all requests (efficient)
model = SentenceTransformer("all-MiniLM-L6-v2")


# ── Core Metric 1: Retrieval Relevance ──────────────────────────────

def score_retrieval_relevance(question: str, chunk: str) -> float:
    """
    Scores how relevant a retrieved chunk is to the question.

    Method: cosine similarity between question and chunk embeddings.
    Range: 0.0 (completely irrelevant) to 1.0 (perfectly relevant)

    Interview explanation:
    "I embed both the question and the chunk using the same model,
    then measure the cosine similarity between their vectors.
    A score above 0.7 means the chunk is directly relevant."
    """
    if not question.strip() or not chunk.strip():
        return 0.0

    # Encode both texts to vectors
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embedding = model.encode(chunk, convert_to_tensor=True)

    # Cosine similarity — returns tensor, .item() converts to float
    similarity = util.cos_sim(question_embedding, chunk_embedding).item()

    # Clamp between 0 and 1 (cosine can technically return negative)
    return round(max(0.0, min(1.0, similarity)), 3)


def score_multiple_chunks(question: str, chunks: list[str]) -> dict:
    """
    Scores all retrieved chunks at once.
    More efficient than calling score_retrieval_relevance in a loop
    because we encode the question only once.

    Returns:
        individual_scores : score per chunk
        overall_score     : average across all chunks
        best_score        : highest scoring chunk's score
        worst_score       : lowest scoring chunk's score
        verdict           : human readable diagnosis
    """
    if not chunks:
        return {
            "individual_scores": [],
            "overall_score": 0.0,
            "best_score": 0.0,
            "worst_score": 0.0,
            "verdict": "No chunks retrieved",
            "diagnosis": "Upload a document first"
        }

    # Encode question once — reuse for all chunks (efficient)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Encode all chunks at once — batch processing is faster
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Score each chunk
    scores = []
    for chunk_emb in chunk_embeddings:
        similarity = util.cos_sim(question_embedding, chunk_emb).item()
        scores.append(round(max(0.0, min(1.0, similarity)), 3))

    overall = round(sum(scores) / len(scores), 3)
    best = max(scores)
    worst = min(scores)

    verdict, diagnosis = get_verdict(overall, best, worst)

    return {
        "individual_scores": scores,
        "overall_score": overall,
        "best_score": best,
        "worst_score": worst,
        "verdict": verdict,
        "diagnosis": diagnosis
    }


# ── Verdict Engine ───────────────────────────────────────────────────

def get_verdict(overall: float, best: float, worst: float) -> tuple[str, str]:
    """
    Converts scores into a human-readable verdict and diagnosis.

    This is the 'debugger' part of RAG Quality Evaluator.
    Instead of just showing numbers, we tell the user what went wrong.

    Interview talking point:
    "I didn't just return scores — I added a diagnosis layer that tells
    you exactly what to fix. That's the difference between a metric
    and a debugger."
    """

    if overall >= 0.7:
        verdict = "Good"
        diagnosis = "Retrieval is working well. Chunks are highly relevant to the question."

    elif overall >= 0.5:
        verdict = "Average"
        if best >= 0.7 and worst < 0.4:
            diagnosis = (
                "Mixed retrieval — some chunks are relevant but others are not. "
                "Try reducing top_k or increasing chunk overlap."
            )
        else:
            diagnosis = (
                "Chunks are partially relevant. "
                "Consider using a better embedding model or smaller chunk size."
            )

    elif overall >= 0.3:
        verdict = "Poor"
        diagnosis = (
            "Retrieval is struggling. The question may be too specific "
            "or the document may not contain relevant information. "
            "Try rephrasing the question or checking your chunk size."
        )

    else:
        verdict = "Critical"
        diagnosis = (
            "Retrieval has failed. Chunks are unrelated to the question. "
            "Possible causes: wrong document uploaded, chunk size too large, "
            "or embedding model mismatch."
        )

    return verdict, diagnosis


# ── Utility: Explain score in plain English ──────────────────────────

def explain_score(score: float) -> str:
    """
    Converts a float score to a plain English label.
    Used in API responses and dashboard tooltips.
    """
    if score >= 0.7:
        return f"{score} — Highly relevant"
    elif score >= 0.5:
        return f"{score} — Moderately relevant"
    elif score >= 0.3:
        return f"{score} — Weakly relevant"
    else:
        return f"{score} — Not relevant"
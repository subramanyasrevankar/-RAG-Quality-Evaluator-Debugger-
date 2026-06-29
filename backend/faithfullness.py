from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def score_faithfulness_local(answer: str, chunks: list) -> dict:
    """
    Scores faithfulness locally using cosine similarity.

    Logic:
    - Embed the answer
    - Embed each chunk
    - Find the highest similarity between answer and any chunk
    - High similarity = answer is grounded in the chunks

    This is fast (no API call) and free.
    Used as a baseline before the Gemini faithfulness check.
    """
    if not answer.strip() or not chunks:
        return {
            "faithfulness_score": 0.0,
            "best_matching_chunk": None,
            "verdict": "No answer or chunks provided"
        }

    answer_embedding = model.encode(answer, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Score answer against each chunk
    scores = [
        round(max(0.0, min(1.0, util.cos_sim(answer_embedding, c).item())), 3)
        for c in chunk_embeddings
    ]

    best_score = max(scores)
    best_chunk_index = scores.index(best_score)

    verdict = get_faithfulness_verdict(best_score)

    return {
        "faithfulness_score": best_score,
        "individual_chunk_scores": scores,
        "best_matching_chunk": chunks[best_chunk_index],
        "best_chunk_index": best_chunk_index,
        "verdict": verdict
    }


def score_context_utilization(answer: str, chunks: list) -> dict:
    """
    Scores how much of the retrieved context was actually used.

    Logic:
    - For each chunk, check if the answer semantically covers it
    - High utilization = answer draws from multiple chunks
    - Low utilization = answer ignores most retrieved context

    Interview talking point:
    'Context utilization is different from faithfulness.
    Faithfulness checks if the answer is grounded in chunks.
    Utilization checks if all retrieved chunks contributed to the answer.
    Low utilization means we retrieved too many irrelevant chunks.'
    """
    if not answer.strip() or not chunks:
        return {
            "utilization_score": 0.0,
            "chunks_used": 0,
            "total_chunks": 0,
            "verdict": "No answer or chunks provided"
        }

    answer_embedding = model.encode(answer, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # A chunk is "used" if similarity with answer is above threshold
    UTILIZATION_THRESHOLD = 0.4
    used_chunks = 0
    chunk_scores = []

    for chunk_emb in chunk_embeddings:
        score = util.cos_sim(answer_embedding, chunk_emb).item()
        score = round(max(0.0, min(1.0, score)), 3)
        chunk_scores.append(score)
        if score >= UTILIZATION_THRESHOLD:
            used_chunks += 1

    utilization_score = round(used_chunks / len(chunks), 3)

    return {
        "utilization_score": utilization_score,
        "chunk_scores": chunk_scores,
        "chunks_used": used_chunks,
        "total_chunks": len(chunks),
        "verdict": get_utilization_verdict(utilization_score)
    }


def combine_scores(
    retrieval_score: float,
    faithfulness_score: float,
    utilization_score: float
) -> dict:
    """
    Combines all three metric scores into one overall RAG score.

    Weights:
    - Retrieval relevance : 40% (most important — garbage in, garbage out)
    - Answer faithfulness : 40% (are we hallucinating?)
    - Context utilization : 20% (are we using what we retrieved?)

    Interview talking point:
    'I weighted retrieval and faithfulness equally at 40% each
    because they represent the two biggest failure modes in RAG —
    bad retrieval and hallucination. Utilization at 20% is a
    secondary signal.'
    """
    overall = round(
        (retrieval_score * 0.4) +
        (faithfulness_score * 0.4) +
        (utilization_score * 0.2),
        3
    )

    return {
        "retrieval_score": retrieval_score,
        "faithfulness_score": faithfulness_score,
        "utilization_score": utilization_score,
        "overall_score": overall,
        "grade": get_grade(overall)
    }


def get_faithfulness_verdict(score: float) -> str:
    if score >= 0.7:
        return "Faithful — answer is well grounded in the context"
    elif score >= 0.5:
        return "Mostly faithful — minor claims may lack support"
    elif score >= 0.3:
        return "Partially faithful — some hallucination detected"
    else:
        return "Unfaithful — answer is not grounded in retrieved chunks"


def get_utilization_verdict(score: float) -> str:
    if score >= 0.7:
        return "High utilization — most chunks contributed to the answer"
    elif score >= 0.4:
        return "Medium utilization — some chunks were ignored"
    else:
        return "Low utilization — retrieved chunks were mostly unused"


def get_grade(overall: float) -> str:
    if overall >= 0.8:
        return "A — Excellent RAG pipeline"
    elif overall >= 0.6:
        return "B — Good RAG pipeline"
    elif overall >= 0.4:
        return "C — Average RAG pipeline"
    elif overall >= 0.2:
        return "D — Poor RAG pipeline"
    else:
        return "F — RAG pipeline is failing"
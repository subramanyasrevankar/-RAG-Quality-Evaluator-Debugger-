from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def score_retrieval_relevance(question: str, chunk: str) -> float:
    if not question.strip() or not chunk.strip():
        return 0.0
    q_emb = model.encode(question, convert_to_tensor=True)
    c_emb = model.encode(chunk, convert_to_tensor=True)
    score = util.cos_sim(q_emb, c_emb).item()
    return round(max(0.0, min(1.0, score)), 3)


def score_multiple_chunks(question: str, chunks: list) -> dict:
    if not chunks:
        return {
            "individual_scores": [],
            "overall_score": 0.0,
            "best_score": 0.0,
            "worst_score": 0.0,
            "verdict": "No chunks retrieved",
            "diagnosis": "Upload a document first"
        }

    q_emb = model.encode(question, convert_to_tensor=True)
    c_embs = model.encode(chunks, convert_to_tensor=True)

    scores = [
        round(max(0.0, min(1.0, util.cos_sim(q_emb, c).item())), 3)
        for c in c_embs
    ]

    overall = round(sum(scores) / len(scores), 3)
    verdict, diagnosis = get_verdict(overall, max(scores), min(scores))

    return {
        "individual_scores": scores,
        "overall_score": overall,
        "best_score": max(scores),
        "worst_score": min(scores),
        "verdict": verdict,
        "diagnosis": diagnosis
    }


def get_verdict(overall: float, best: float, worst: float):
    if overall >= 0.7:
        return "Good", "Retrieval is working well. Chunks are highly relevant."
    elif overall >= 0.5:
        if best >= 0.7 and worst < 0.4:
            return "Average", "Mixed retrieval — some chunks relevant, others not. Try reducing top_k or increasing overlap."
        return "Average", "Chunks are partially relevant. Consider smaller chunk size."
    elif overall >= 0.3:
        return "Poor", "Retrieval is struggling. Try rephrasing the question or checking chunk size."
    else:
        return "Critical", "Retrieval has failed. Wrong document uploaded or chunk size too large."


def explain_score(score: float) -> str:
    if score >= 0.7:
        return f"{score} — Highly relevant"
    elif score >= 0.5:
        return f"{score} — Moderately relevant"
    elif score >= 0.3:
        return f"{score} — Weakly relevant"
    return f"{score} — Not relevant"
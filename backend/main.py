from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.chunker import chunk_document, get_chunk_stats
from backend.retriever import store_chunks, retrieve_chunks
from backend.evaluator import score_multiple_chunks
from backend.llm import generate_answer, check_faithfulness_with_llm
from backend.faithfulness import (
    score_faithfulness_local,
    score_context_utilization,
    combine_scores
)
from backend.database import (
    get_db, init_db, save_evaluation_run,
    save_uploaded_document, get_recent_runs, get_average_scores
)

app = FastAPI(
    title="RAG Quality Evaluator",
    description="Evaluates why your RAG pipeline is failing",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# ── Request/Response models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list
    retrieval_score: float
    faithfulness_score: float
    utilization_score: float
    overall_score: float
    grade: str
    verdict: str
    diagnosis: str
    hallucinated: bool


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Quality Evaluator v2.0 is running"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith((".txt", ".pdf", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .pdf, .md files supported"
        )

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    chunks = chunk_document(text)
    stats = get_chunk_stats(chunks)

    store_chunks(chunks, source=file.filename)

    save_uploaded_document(
        db=db,
        filename=file.filename,
        total_chunks=len(chunks),
        file_size_bytes=len(content)
    )

    return {
        "filename": file.filename,
        "total_chunks": len(chunks),
        "chunk_stats": stats,
        "message": f"Successfully stored {len(chunks)} chunks"
    }


@app.post("/query", response_model=QueryResponse)
def query_document(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1 — Retrieve chunks
    chunks = retrieve_chunks(
        question=request.question,
        top_k=request.top_k
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No chunks found. Upload a document first."
        )

    # Step 2 — Score retrieval relevance
    retrieval_eval = score_multiple_chunks(request.question, chunks)
    retrieval_score = retrieval_eval["overall_score"]

    # Step 3 — Generate answer using Gemini
    answer = generate_answer(request.question, chunks)

    # Step 4 — Score faithfulness (local)
    faith_eval = score_faithfulness_local(answer, chunks)
    faithfulness_score = faith_eval["faithfulness_score"]

    # Step 5 — Score context utilization
    util_eval = score_context_utilization(answer, chunks)
    utilization_score = util_eval["utilization_score"]

    # Step 6 — Check faithfulness with Gemini (LLM-as-judge)
    llm_faith = check_faithfulness_with_llm(request.question, answer, chunks)
    hallucinated = llm_faith["hallucinated"]

    # Blend local + LLM faithfulness scores (50/50)
    blended_faithfulness = round(
        (faithfulness_score + llm_faith["faithfulness_score"]) / 2, 3
    )

    # Step 7 — Combine all scores
    combined = combine_scores(
        retrieval_score=retrieval_score,
        faithfulness_score=blended_faithfulness,
        utilization_score=utilization_score
    )

    # Step 8 — Save to PostgreSQL
    save_evaluation_run(
        db=db,
        question=request.question,
        answer=answer,
        retrieval_score=retrieval_score,
        faithfulness_score=blended_faithfulness,
        utilization_score=utilization_score,
        overall_score=combined["overall_score"],
        verdict=retrieval_eval["verdict"],
        diagnosis=retrieval_eval["diagnosis"],
        chunks_retrieved=len(chunks),
        top_k=request.top_k
    )

    return QueryResponse(
        question=request.question,
        answer=answer,
        retrieved_chunks=chunks,
        retrieval_score=retrieval_score,
        faithfulness_score=blended_faithfulness,
        utilization_score=utilization_score,
        overall_score=combined["overall_score"],
        grade=combined["grade"],
        verdict=retrieval_eval["verdict"],
        diagnosis=retrieval_eval["diagnosis"],
        hallucinated=hallucinated
    )


@app.get("/history")
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    runs = get_recent_runs(db, limit=limit)
    return [
        {
            "id": run.id,
            "question": run.question,
            "answer": run.answer,
            "retrieval_score": run.retrieval_score,
            "faithfulness_score": run.faithfulness_score,
            "utilization_score": run.utilization_score,
            "overall_score": run.overall_score,
            "verdict": run.verdict,
            "created_at": run.created_at
        }
        for run in runs
    ]


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    return get_average_scores(db)
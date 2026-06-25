from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .chunker import chunk_document, get_chunk_stats
from .retriever import store_chunks, retrieve_chunks
from .evaluator import score_multiple_chunks
from .database import (
    get_db,
    init_db,
    save_evaluation_run,
    save_uploaded_document,
    get_recent_runs,
    get_average_scores,
)
app = FastAPI(
    title="RAG Quality Evaluator",
    description="Evaluates why your RAG pipeline is failing",
    version="1.0.0"
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


# ── Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    question: str
    retrieved_chunks: list
    individual_scores: list
    overall_score: float
    verdict: str
    diagnosis: str


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Quality Evaluator API is running"}


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

    chunks = retrieve_chunks(
        question=request.question,
        top_k=request.top_k
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No chunks found. Upload a document first."
        )

    evaluation = score_multiple_chunks(request.question, chunks)

    save_evaluation_run(
        db=db,
        question=request.question,
        retrieval_score=evaluation["overall_score"],
        overall_score=evaluation["overall_score"],
        verdict=evaluation["verdict"],
        diagnosis=evaluation["diagnosis"],
        chunks_retrieved=len(chunks),
        top_k=request.top_k
    )

    return QueryResponse(
        question=request.question,
        retrieved_chunks=chunks,
        individual_scores=evaluation["individual_scores"],
        overall_score=evaluation["overall_score"],
        verdict=evaluation["verdict"],
        diagnosis=evaluation["diagnosis"]
    )


@app.get("/history")
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    runs = get_recent_runs(db, limit=limit)
    return [
        {
            "id": run.id,
            "question": run.question,
            "retrieval_score": run.retrieval_score,
            "overall_score": run.overall_score,
            "verdict": run.verdict,
            "created_at": run.created_at
        }
        for run in runs
    ]


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    return get_average_scores(db)
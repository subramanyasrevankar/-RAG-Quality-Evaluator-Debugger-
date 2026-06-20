from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.chunker import chunk_document
from backend.retriever import store_chunks, retrieve_chunks
from backend.evaluator import score_retrieval_relevance

app=FastAPI(
    title="Rag quality evaluation API",
    description="Evaluates why your RAG pipeline is failing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query:str
    top_k:int=3


class EvaluationRequest(BaseModel):
    query:str
    retrieved_chunks:list[str]
    relevant_score:list[float]
    overall_score:float
    verdict:str

app.get("/")
def home():
    return {"message":"Welcome to the RAG quality evaluation API. Use the /evaluate endpoint to evaluate your RAG pipeline."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document. It gets chunked and stored
    in ChromaDB as embeddings.
    """
    if not file.filename.endswith((".txt", ".pdf", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .pdf, .md files supported in Phase 1"
        ) 
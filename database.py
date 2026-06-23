from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
# create engine to connect to the SQLite database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 
from datetime import datetime 
import os 
from dotenv import load_dotenv

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/rag_evaluator"
)

# if it does not contains the database url then it will default to a local SQLite database file named rag_evaluator.db in the current directory
engine=create_engine(DATABASE_URL)
session_local=sessionmaker(autocommit=False,autoflush=False,bind=engine)
# autoflush helps in the synchronisation,bind=engine helps to chose the database engine to connect to,autocommit=False means that changes to the database will not be automatically committed after each operation, allowing for more control over when changes are saved.

class Evolutionrun(Base):
    __tablename__="evaluation_runs"
    id=Column(Integer,primary_key=True,index=True)
    question=Column(Text,nullable=False)
    answer=Column(Text,nullable=False)
    Source=Column(String,nullable=False)


        # Evaluation scores (Phase 1 has retrieval only)
    retrieval_score     = Column(Float, nullable=True)
    faithfulness_score  = Column(Float, nullable=True)      # added in Phase 2
    utilization_score   = Column(Float, nullable=True)      # added in Phase 2
    overall_score       = Column(Float, nullable=True)



class UploadedDocument(Base):
    """
    Tracks every document uploaded to the system.

    Useful for the dashboard to show per-document performance
    and for filtering evaluation runs by document.
    """
    __tablename__ = "uploaded_documents"

    id              = Column(Integer, primary_key=True, index=True)
    filename        = Column(String(255), nullable=False)
    total_chunks    = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    uploaded_at     = Column(DateTime, default=datetime.utcnow)


def init_db():  
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
    

def get_db():
    db=session_local()  
    try:
        yield db 
    finally:
        db.close()  
        # giving the control back to the caller after the database session is closed, ensuring that resources are released properly.    


def save_evaluation_run(
    db,
    question: str,
    retrieval_score: float,
    overall_score: float,
    verdict: str,
    diagnosis: str,
    source_document: str = None,
    top_k: int = 3,
    chunks_retrieved: int = 0
) -> EvaluationRun:
    """
    Saves a completed evaluation run to PostgreSQL.
    Called from the /query endpoint after scoring.
    """
    run = Evolutionrun(
        question=question,
        source_document=source_document,
        retrieval_score=retrieval_score,
        overall_score=overall_score,
        verdict=verdict,
        diagnosis=diagnosis,
        top_k=top_k,
        chunks_retrieved=chunks_retrieved
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def save_uploaded_document(
    db,
    filename: str,
    total_chunks: int,
    file_size_bytes: int = 0
) -> UploadedDocument:
    """
    Saves document upload metadata to PostgreSQL.
    Called from the /upload endpoint after chunking.
    """
    doc = UploadedDocument(
        filename=filename,
        total_chunks=total_chunks,
        file_size_bytes=file_size_bytes
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc



def get_recent_runs(db, limit: int = 10) -> list[Evolutionrun]:
    """
    Fetches most recent evaluation runs.
    Powers the history view in the Phase 3 dashboard.
    """
    return (
        db.query(Evolutionrun)
        .order_by(Evolutionrun.created_at.desc())
        .limit(limit)
        .all()
    )


def get_average_scores(db) -> dict:
    """
    Returns average scores across all runs.
    Powers the summary stats on the dashboard.
    """
    from sqlalchemy import func

    result = db.query(
        func.avg(Evolutionrun.retrieval_score).label("avg_retrieval"),
        func.avg(Evolutionrun.overall_score).label("avg_overall"),
        func.count(Evolutionrun.id).label("total_runs")
    ).first()

    return {
        "avg_retrieval_score": round(result.avg_retrieval or 0, 3),
        "avg_overall_score": round(result.avg_overall or 0, 3),
        "total_runs": result.total_runs or 0
    }

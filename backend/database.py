from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/rag_evaluator"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id                  = Column(Integer, primary_key=True, index=True)
    question            = Column(Text, nullable=False)
    source_document     = Column(String(255), nullable=True)
    answer              = Column(Text, nullable=True)
    retrieval_score     = Column(Float, nullable=True)
    faithfulness_score  = Column(Float, nullable=True)
    utilization_score   = Column(Float, nullable=True)
    overall_score       = Column(Float, nullable=True)
    verdict             = Column(String(50), nullable=True)
    diagnosis           = Column(Text, nullable=True)
    top_k               = Column(Integer, default=3)
    chunks_retrieved    = Column(Integer, default=0)
    created_at          = Column(DateTime, default=datetime.utcnow)


class UploadedDocument(Base):
    __tablename__ = "uploaded_documents"

    id              = Column(Integer, primary_key=True, index=True)
    filename        = Column(String(255), nullable=False)
    total_chunks    = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    uploaded_at     = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_evaluation_run(db, question, retrieval_score, overall_score,
                        verdict, diagnosis, source_document=None,
                        top_k=3, chunks_retrieved=0):
    run = EvaluationRun(
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


def save_uploaded_document(db, filename, total_chunks, file_size_bytes=0):
    doc = UploadedDocument(
        filename=filename,
        total_chunks=total_chunks,
        file_size_bytes=file_size_bytes
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_recent_runs(db, limit=10):
    return (
        db.query(EvaluationRun)
        .order_by(EvaluationRun.created_at.desc())
        .limit(limit)
        .all()
    )


def get_average_scores(db):
    result = db.query(
        func.avg(EvaluationRun.retrieval_score).label("avg_retrieval"),
        func.avg(EvaluationRun.overall_score).label("avg_overall"),
        func.count(EvaluationRun.id).label("total_runs")
    ).first()

    return {
        "avg_retrieval_score": round(result.avg_retrieval or 0, 3),
        "avg_overall_score": round(result.avg_overall or 0, 3),
        "total_runs": result.total_runs or 0
    }
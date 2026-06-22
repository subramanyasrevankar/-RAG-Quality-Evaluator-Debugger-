cat > README.md << 'EOF'
# RAG Quality Evaluator & Debugger

A tool that evaluates **why your RAG pipeline is failing** and tells you exactly which step broke — retrieval, generation, or context usage.

## The Problem
Every team building RAG hits the same wall: bad answers, no idea where it broke. Was it the retrieval? The LLM? The chunk size? This tool gives you per-step scores.

## Evaluation Metrics
| Metric | What it measures |
|---|---|
| Retrieval Relevance | Are the retrieved chunks actually related to the question? |
| Answer Faithfulness | Does the answer stick to what the chunks say? |
| Context Utilization | How much of the retrieved context was actually used? |

## Tech Stack
- **Backend**: FastAPI + PostgreSQL + ChromaDB
- **Embeddings**: Sentence Transformers
- **LLM**: Claude API (Anthropic)
- **Cache**: Redis
- **Frontend**: React + Recharts

## Phases
- [x] Phase 1 — Core RAG pipeline + retrieval relevance scorer
- [ ] Phase 2 — Claude integration + faithfulness evaluator
- [ ] Phase 3 — React dashboard + analytics
- [ ] Phase 4 — Redis caching + semantic similarity cache
- [ ] Phase 5 — Debug suggestions + export

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```
EOF

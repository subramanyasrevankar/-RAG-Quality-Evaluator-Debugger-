import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(chunks: list, source: str = "unknown") -> None:
    if not chunks:
        return

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    ids = [f"{source}_{timestamp}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source, "chunk_index": i, "timestamp": timestamp}
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )


def retrieve_chunks(question: str, top_k: int = 3) -> list:
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count())
    )
    return results["documents"][0]


def retrieve_chunks_with_scores(question: str, top_k: int = 3) -> list:
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"]
    )

    output = []
    for doc, distance, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        output.append({
            "text": doc,
            "similarity_score": round(1 - distance, 3),
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1)
        })

    return output


def clear_collection() -> None:
    global collection
    client.delete_collection("documents")
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
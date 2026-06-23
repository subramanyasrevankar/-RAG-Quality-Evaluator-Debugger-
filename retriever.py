import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

# needs to have an embedding function that takes in a list of strings and returns a list of vectors 
client=chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# get or create the collection where will store the document

collection=client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(chunks:list[str],source:str="unknown"):
    """
    Stores chunks in ChromaDB as embeddings.

    Each chunk gets:
    - A unique ID
    - The text itself
    - Metadata (source filename, timestamp, chunk index)

    Metadata is useful later for filtering by document.
    """
    if not chunks:
        return 
     
    timestamp=datetime.now().strftime("%Y%m%d%H%M%S")
    ids=[f"{source}_{timestamp}_{i} " for i in range(len(chunks))]
    # embedding_function will automatically be applied to the chunks when we add them to the collection
    metadatas=[
        {
            "source":source,
            "chunk_index":i,
            "timestamp":timestamp
        }
        for i in range(len(chunks))
    ]

    # add to collection 
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks from '{source}' in ChromaDB")

    # chroma also stores the embeddings in the collection, but we don't need to manage that part directly — it happens under the hood when we add documents with an embedding function defined for the collection. 


def retrieve_chunks(query:str,top_k:int=3)->list[str]:
     if collection.count()==0:
         return []
     results=collection.query(
         query_texts=[query],
         n_results=min(top_k,collection.count())
     )
    #  results is a dict with keys: ids, documents, metadatas, distances
      
     chunks=results["documents"][0]


def retrieve_chunks_with_score(question:str,top_k:int=3)->list[dict]:
    if collection.count()==0 :
        return []
    results=collection.query(
        query_texts=[question],
        n_results=min(top_k,collection.count()),
        include=["document","distances","metadatas"]
    )

    # the collection contains all the necessary things.
    chunks_with_scores=[]
    documents=results["documents"][0]
    distances=results["distances"][0]
    metadatas=results["metadatas"][0]

    for doc, distance, meta in zip(documents, distances, metadatas):
        # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
        # Convert to similarity score (1 = identical, 0 = opposite)
        similarity = round(1 - distance, 3)

        chunks_with_scores.append({
            "text": doc,
            "similarity_score": similarity,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1)
        })

    return chunks_with_scores



def clear_collection()->None:
    """ Clears all chunks from ChromaDB. Useful for testing — also added as DELETE /reset endpoint later. """
    global collection 
    client.delete_collection("documents")
    collection=client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print("ChromaDb collection cleared")

import chromadb
from ragatouille import RAGPretrainedModel

def query_database(question, collection_name="medical_qa_collection", persist_path="/path/to/persist", n_results=5, rerank=False, num_docs_final=3):
    # Initialize Chroma client and query for relevant documents
    chroma_client = chromadb.PersistentClient(path=persist_path)
    collection = chroma_client.get_collection(name=collection_name)
    
    # Query Chroma collection based on the input question
    results = collection.query(query_texts=[question], n_results=n_results)
    relevant_docs = results["documents"]

    # Load the reranker model (if reranking is enabled)
    reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0") if rerank else None
    
    # Apply reranking if the reranker is provided
    if reranker:
        reranked_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        # Extract the final content of the reranked documents
        relevant_docs = [doc["content"] for doc in reranked_docs]
    
    return relevant_docs




from datasets import load_dataset
import chromadb
import pandas as pd

def setup_chroma_db(collection_name="medical_qa_collection", max_rows=15000, persist_path = ""):
    
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
    data = data.to_pandas()
    data["id"] = data.index
    
    
    subset_data = data.head(max_rows)
    
    
    chroma_client = chromadb.PersistentClient(path=persist_path)
    
    
    if collection_name in [coll.name for coll in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)
    
   
    collection = chroma_client.create_collection(name=collection_name)
    collection.add(
        documents=subset_data["Answer"].tolist(),
        metadatas=[{"qtype": qtype} for qtype in subset_data["qtype"].tolist()],
        ids=[f"id{x}" for x in range(max_rows)]
    )
    print(f"Chroma DB collection '{collection_name}' created with {max_rows} documents.")

if __name__ == "__main__":
    setup_chroma_db()


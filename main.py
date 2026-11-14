from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("./notebook/data")
    store = FaissVectorStore("faiss_store")
    if os.path.exists(os.path.join("faiss_store", "faiss.index")):
        store.load()
    else:
        store.build_from_documents(docs)
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What were the quantitative metrics for evaluation of computational ai artists?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
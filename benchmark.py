import time
import random
from llm_analyzer import classify_intent, SemanticRouter
from rag_processor import chunk_text, cosine_similarity

def test_router():
    queries = [
        "What is the stock price of TCS?",
        "TCS",
        "RELIANCE.NS",
        "tell me about WIPRO",
        "How do mutual funds work?",
        "What is a PE ratio?",
        "Who is the CEO of Apple?",
        "HDFC bank stock report"
    ] * 1000  # 8000 queries total

    start_time = time.time()
    for q in queries:
        _looks_like_stock_query(q)
    end_time = time.time()

    print(f"--- Intent Router Benchmarks ---")
    print(f"Total queries processed: {len(queries)}")
    print(f"Total time: {(end_time - start_time):.4f} seconds")
    print(f"Average time per query: {((end_time - start_time) / len(queries)) * 1000:.4f} ms")


def test_chunking():
    # Generate 500 pages of text (~1.5M characters)
    dummy_text = "This is a dummy sentence to simulate document text for chunking purposes. It has some length. " * 15000
    
    start_time = time.time()
    chunks = chunk_text(dummy_text, chunk_size=1000, chunk_overlap=200)
    end_time = time.time()
    
    print(f"\n--- Chunking Benchmarks ---")
    print(f"Text length: {len(dummy_text)} characters")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Total time to chunk: {(end_time - start_time):.4f} seconds")


def test_vector_search():
    num_vectors = 10000
    dim = 768
    
    print(f"\n--- Vector Search Benchmarks ({num_vectors} embeddings) ---")
    print("Generating random vectors (simulating embeddings)...")
    
    # Generate random query and database vectors
    query_vector = [random.random() for _ in range(dim)]
    db_vectors = [[random.random() for _ in range(dim)] for _ in range(num_vectors)]
    
    start_time = time.time()
    
    # Simulate similarity_search scoring loop
    scored_chunks = []
    for i, emb in enumerate(db_vectors):
        score = cosine_similarity(query_vector, emb)
        scored_chunks.append((i, score))
        
    # Sort
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_k = scored_chunks[:3]
    
    end_time = time.time()
    
    print(f"Total time for search and sort over {num_vectors} vectors: {(end_time - start_time):.4f} seconds")
    print(f"Latency per query: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    test_router()
    test_chunking()
    test_vector_search()

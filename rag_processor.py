# rag_processor.py - Clean, Lightweight Native Python RAG Engine

import os
import requests
import re
from pypdf import PdfReader

# --- Configuration ---
EMBEDDING_MODEL = "models/gemini-embedding-2"
FALLBACK_MODEL = "models/gemini-embedding-001"


def extract_text_from_pdf(pdf_file) -> str:
    """Extracts clean text from a Streamlit uploaded PDF file object."""
    try:
        reader = PdfReader(pdf_file)
        text_parts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")

def extract_text_from_txt(txt_file) -> str:
    """Extracts text from a Streamlit uploaded TXT file object."""
    try:
        return txt_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise ValueError(f"Failed to parse TXT: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits text into overlapping character-level chunks.
    Ensures that sentence/paragraph structures are generally maintained.
    """
    if not text:
        return []
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # If we are not at the end of the text, try to find a nice boundary
        if end < text_len:
            # Look for a sentence ending, paragraph, or space
            boundary = -1
            for char in ['. ', '? ', '! ', '\n', ' ']:
                idx = text.rfind(char, start, end)
                if idx > start + (chunk_size // 2):
                    boundary = idx + len(char)
                    break
            if boundary != -1:
                end = boundary
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        if end >= text_len:
            break
            
        start = end - chunk_overlap
        
    return chunks

def _get_embedding_rest(text: str, api_key: str, model: str) -> list[float]:
    """Helper to fetch a single embedding vector via the Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={api_key}"
    payload = {
        "model": model,
        "content": {
            "parts": [{"text": text}]
        }
    }
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Fetches embeddings for a list of texts.
    Leverages Gemini REST batchEmbedContents to run highly efficiently in a single round-trip.
    Falls back to single requests if batching fails or the model is unavailable.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in your environment/dotenv.")
        
    if not texts:
        return []
        
    # Attempt batch embedding
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL}:batchEmbedContents?key={api_key}"
    
    requests_payload = []
    for text in texts:
        requests_payload.append({
            "model": EMBEDDING_MODEL,
            "content": {"parts": [{"text": text}]}
        })
        
    payload = {"requests": requests_payload}
    
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        return [emb["values"] for emb in embeddings]
    except Exception as e:
        # Fallback to single requests with fallback model if required
        results = []
        for text in texts:
            try:
                results.append(_get_embedding_rest(text, api_key, EMBEDDING_MODEL))
            except Exception:
                # Absolute fallback model
                results.append(_get_embedding_rest(text, api_key, FALLBACK_MODEL))
        return results

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculates the cosine similarity between two vector lists."""
    dot = sum(x * y for x, y in zip(v1, v2))
    mag1 = sum(x * x for x in v1) ** 0.5
    mag2 = sum(y * y for y in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

class SimpleVectorStore:
    """A lightweight, in-memory vector database using cosine similarity."""
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        
    def add_documents(self, texts: list[str], embeddings: list[list[float]]):
        self.chunks.extend(texts)
        self.embeddings.extend(embeddings)
        
    def similarity_search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Queries the vector index using cosine similarity and returns top_k matches."""
        if not self.chunks or not self.embeddings:
            return []
            
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return []
            
        try:
            query_vector = _get_embedding_rest(query, api_key, EMBEDDING_MODEL)
        except Exception:
            try:
                query_vector = _get_embedding_rest(query, api_key, FALLBACK_MODEL)
            except Exception:
                return []
                
        scored_chunks = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            score = cosine_similarity(query_vector, emb)
            scored_chunks.append((chunk, score))
            
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

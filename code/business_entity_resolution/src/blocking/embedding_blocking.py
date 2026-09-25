import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

_model = None

def _get_model():
    global _model
    if _model is None and HAS_ST:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _encode_batch(texts: list, batch_size: int = 256) -> np.ndarray:
    model = _get_model()
    if model is None:
        return np.zeros((len(texts), 1), dtype=np.float32)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.astype(np.float32)

def embedding_block(s1_df: pd.DataFrame, pool_df: pd.DataFrame, max_candidates: int = 30) -> dict:
    """
    Encode 'business_name + business_address' for S1 and pool.
    Use FAISS for approximate nearest neighbor search.
    Returns dict[s1_entity_id -> set[pool_entity_id]].
    If sentence-transformers or faiss unavailable, returns empty sets.
    """
    if not HAS_ST or not HAS_FAISS:
        # Graceful fallback: return empty candidates
        return {row["entity_id"]: set() for _, row in s1_df.iterrows()}
    
    print("    Encoding pool embeddings...")
    pool_ids = pool_df["entity_id"].tolist()
    pool_texts = [
        f"{str(row.get('business_name', ''))} {str(row.get('business_address', ''))}".strip()
        for _, row in pool_df.iterrows()
    ]
    pool_embs = _encode_batch(pool_texts)
    
    # Build FAISS index
    dim = pool_embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine (embeddings are normalized)
    index.add(pool_embs)
    
    print("    Encoding S1 embeddings...")
    s1_ids = s1_df["entity_id"].tolist()
    s1_texts = [
        f"{str(row.get('business_name', ''))} {str(row.get('business_address', ''))}".strip()
        for _, row in s1_df.iterrows()
    ]
    s1_embs = _encode_batch(s1_texts)
    
    # Search
    k = min(max_candidates, len(pool_ids))
    print(f"    FAISS search k={k}...")
    distances, indices = index.search(s1_embs, k)
    
    result = {}
    for i, sid in enumerate(s1_ids):
        cands = set()
        for j in range(k):
            idx = indices[i][j]
            if idx >= 0 and idx < len(pool_ids):
                cands.add(pool_ids[idx])
        result[sid] = cands
    
    return result

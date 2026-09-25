import math
from collections import Counter
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

_model = None
_cache = {}

def get_char_ngrams(text, n=4):
    return [text[i:i+n] for i in range(len(text)-n+1)] if len(text) >= n else [text] if text else []

def _tfidf_cosine(s1, s2):
    if not s1 or not s2:
        return 0.0
    c1 = Counter(get_char_ngrams(s1, 4))
    c2 = Counter(get_char_ngrams(s2, 4))
    intersection = set(c1.keys()) & set(c2.keys())
    # TF-IDF approx with just TF here since we don't have global IDF
    numerator = sum([c1[x] * c2[x] for x in intersection])
    sum1 = sum([c1[x]**2 for x in c1.keys()])
    sum2 = sum([c2[x]**2 for x in c2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator / denominator) if denominator else 0.0

def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def _get_embedding(text):
    global _model
    if not text:
        return None
    if text in _cache:
        return _cache[text]
    if _model is None and HAS_ST:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    if _model:
        emb = _model.encode(text)
        _cache[text] = emb
        return emb
    return None

def compute_semantic_features(s1_row, s2_row) -> dict[str, float]:
    f = {}
    name1 = str(s1_row.get('business_name', ''))
    name2 = str(s2_row.get('business_name', ''))
    addr1 = str(s1_row.get('business_address', ''))
    addr2 = str(s2_row.get('business_address', ''))
    full1 = f"{name1} {addr1}".strip()
    full2 = f"{name2} {addr2}".strip()
    
    if HAS_ST:
        f['semantic_name_cosine'] = cosine_sim(_get_embedding(name1), _get_embedding(name2))
        f['semantic_addr_cosine'] = cosine_sim(_get_embedding(addr1), _get_embedding(addr2))
        f['semantic_full_cosine'] = cosine_sim(_get_embedding(full1), _get_embedding(full2))
        f['has_semantic'] = 1.0
    else:
        # fallback: TF-IDF char ngram
        f['semantic_name_cosine'] = _tfidf_cosine(name1, name2)
        f['semantic_addr_cosine'] = _tfidf_cosine(addr1, addr2)
        f['semantic_full_cosine'] = _tfidf_cosine(full1, full2)
        f['has_semantic'] = 0.0
    return f

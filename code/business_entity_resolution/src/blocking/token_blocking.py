from collections import defaultdict, Counter
import math
import pandas as pd
from src.normalization import normalize_text, name_without_suffix, tokens

MAX_POSTING = 5000
RARE_K = 6

def token_block(s1_df: pd.DataFrame, pool_df: pd.DataFrame, max_candidates: int = 50) -> dict:
    """
    Build inverted index on pool name tokens (without legal suffix).
    For each S1 record, retrieve candidates sharing rare name tokens,
    weighted by IDF. Also uses character 3-grams for typo tolerance.
    Returns dict[s1_entity_id -> set[pool_entity_id]]
    """
    pool_ids = pool_df["entity_id"].tolist()
    pool_names = pool_df["business_name"].fillna("").tolist()
    
    # Build token -> set(pool_ids) inverted index
    inv = defaultdict(set)
    char3_inv = defaultdict(set)  # char 3-gram index for typo tolerance
    for pid, name in zip(pool_ids, pool_names):
        norm = name_without_suffix(normalize_text(name))
        toks = norm.split()
        for t in set(toks):
            if t and len(t) >= 2:
                inv[t].add(pid)
            # char 3-grams for longer tokens
            if len(t) >= 4:
                for i in range(len(t) - 2):
                    char3_inv[t[i:i+3]].add(pid)
    
    # Filter huge posting lists
    inv = {k: v for k, v in inv.items() if len(v) <= MAX_POSTING}
    char3_inv = {k: v for k, v in char3_inv.items() if len(v) <= MAX_POSTING}
    
    # IDF weights
    N = max(len(pool_ids), 1)
    idf = {t: math.log(N / (len(pids) + 1)) for t, pids in inv.items()}
    
    result = {}
    for _, row in s1_df.iterrows():
        sid = row["entity_id"]
        norm = name_without_suffix(normalize_text(row.get("business_name", "") or ""))
        toks = norm.split()
        if not toks:
            result[sid] = set()
            continue
        
        # Sort tokens by IDF (rarest first)
        tok_idf = [(t, idf.get(t, 0)) for t in toks if t in inv]
        tok_idf.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in tok_idf[:RARE_K]]
        
        # Score candidates by sum of shared token IDFs
        cand_scores = Counter()
        for t in selected:
            w = idf.get(t, 1.0)
            for pid in inv[t]:
                cand_scores[pid] += w
        
        # Also check char 3-grams for tokens not found in index
        unfound = [t for t in toks if t not in inv and len(t) >= 4]
        for t in unfound[:3]:  # limit to avoid explosion
            for i in range(len(t) - 2):
                gram = t[i:i+3]
                if gram in char3_inv:
                    for pid in char3_inv[gram]:
                        cand_scores[pid] += 0.5
        
        top = set(pid for pid, _ in cand_scores.most_common(max_candidates))
        result[sid] = top
    
    return result
